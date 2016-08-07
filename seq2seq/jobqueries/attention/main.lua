require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require('pl.stringx').import()
require 'pl.seq'
require 'utils/SymbolsManager'
include "../utils/utils.lua"
local MinibatchLoader = require 'utils.MinibatchLoader'

function transfer_data(x)
  if opt.gpuid>=0 then
    return x:cuda()
  end
  return x
end

function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(opt.rnn_size, 4 * opt.rnn_size)(x)
  local h2h = nn.Linear(opt.rnn_size, 4 * opt.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension
  local reshaped_gates =  nn.Reshape(4, opt.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates)):annotate{name='in_gate'}
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates)):annotate{name='in_transform'}
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates)):annotate{name='forget_gate'}
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates)):annotate{name='out_gate'}
  
  if opt.dropoutrec > 0 then
    in_transform = nn.Dropout(opt.dropoutrec)(in_transform)
  end

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}):annotate{name='next_c_1'},
      nn.CMulTable()({in_gate, in_transform}):annotate{name='next_c_2'}
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}):annotate{name='next_h'}

  return next_c, next_h
end

function create_lstm_unit(w_size)
  -- input
  local x = nn.Identity()()
  local prev_s = nn.Identity()()

  local i = {[0] = nn.LookupTable(w_size, opt.rnn_size)(x):annotate{name='lstm'}}
  local next_s = {}
  local splitted = {prev_s:split(2 * opt.num_layers)}
  for layer_idx = 1, opt.num_layers do
    local prev_c = splitted[2 * layer_idx - 1]
    local prev_h = splitted[2 * layer_idx]
    local x_in = i[layer_idx - 1]
    if opt.dropout > 0 then
      x_in = nn.Dropout(opt.dropout)(x_in)
    end
    local next_c, next_h = lstm(x_in, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local m = nn.gModule({x, prev_s}, {nn.Identity()(next_s)})
  
  return transfer_data(m)
end

function create_attention_unit(w_size)
  -- input
  local enc_s_top = nn.Identity()()
  local dec_s_top = nn.Identity()()

  -- (batch*length*H) * (batch*H*1) = (batch*length*1)
  local dot = nn.MM()({enc_s_top, nn.View(opt.rnn_size,1):setNumInputDims(1)(dec_s_top)})
  local attention = nn.SoftMax()(nn.Sum(3)(dot))
  -- (batch*length*H)^T * (batch*length*1) = (batch*H*1)
  local enc_attention = nn.MM(true, false)({enc_s_top, nn.View(-1, 1):setNumInputDims(1)(attention)})
  local hid = nn.Tanh()(nn.Linear(2*opt.rnn_size, opt.rnn_size)(nn.JoinTable(2)({nn.Sum(3)(enc_attention), dec_s_top})))
  local h2y_in = hid
  if opt.dropout > 0 then
    h2y_in = nn.Dropout(opt.dropout)(h2y_in)
  end
  local h2y = nn.Linear(opt.rnn_size, w_size)(h2y_in)
  local pred = nn.LogSoftMax()(h2y)
  local m = nn.gModule({enc_s_top, dec_s_top}, {pred})
  
  return transfer_data(m)
end

function setup()
  -- initialize model
  model = {}
  model.enc_s = {}
  model.dec_s = {}
  model.ds = {}
  model.save_enc_ds = {}
  model.enc_s_top = transfer_data(torch.Tensor(opt.batch_size, opt.enc_seq_length, opt.rnn_size))
  model.enc_ds_top = transfer_data(torch.Tensor(opt.batch_size, opt.enc_seq_length, opt.rnn_size))
  
  for j = 0, opt.enc_seq_length do
    model.enc_s[j] = {}
    for d = 1, 2 * opt.num_layers do
      model.enc_s[j][d] = transfer_data(torch.zeros(opt.batch_size, opt.rnn_size))
    end
  end
  
  for j = 0, opt.dec_seq_length do
    model.dec_s[j] = {}
    for d = 1, 2 * opt.num_layers do
      model.dec_s[j][d] = transfer_data(torch.zeros(opt.batch_size, opt.rnn_size))
    end
  end

  for d = 1, 2 * opt.num_layers do
    model.ds[d] = transfer_data(torch.zeros(opt.batch_size, opt.rnn_size))
    model.save_enc_ds[d] = transfer_data(torch.zeros(opt.batch_size, opt.rnn_size))
  end

  local word_manager, form_manager = unpack(torch.load(path.join(opt.data_dir, 'map.t7')))

  print("Creating encoder")
  model.enc_rnn_unit = create_lstm_unit(word_manager.vocab_size)
  model.enc_rnn_unit:training()

  print("Creating decoder")
  model.dec_rnn_unit = create_lstm_unit(form_manager.vocab_size)
  model.dec_rnn_unit:training()
  model.dec_att_unit = create_attention_unit(form_manager.vocab_size)

  model.criterions={}
  for i = 1, opt.dec_seq_length do
    table.insert(model.criterions, transfer_data(nn.ClassNLLCriterion()))
  end

  -- collect all parameters to a vector
  param_x, param_dx = combine_all_parameters(model.enc_rnn_unit, model.dec_rnn_unit, model.dec_att_unit)
  print('number of parameters in the model: ' .. param_x:nElement())
  
  param_x:uniform(-opt.init_weight, opt.init_weight)

  -- make a bunch of clones after flattening, as that reallocates memory (tips from char-rnn)
  model.enc_rnns = cloneManyTimes(model.enc_rnn_unit, opt.enc_seq_length)
  model.dec_rnns = cloneManyTimes(model.dec_rnn_unit, opt.dec_seq_length)
  model.dec_atts = cloneManyTimes(model.dec_att_unit, opt.dec_seq_length)
end

function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

function eval_training(param_x_)
  model.enc_rnn_unit:training()
  model.dec_rnn_unit:training()
  model.dec_att_unit:training()
  for i = 1, #model.enc_rnns do
    model.enc_rnns[i]:training()
  end
  for i = 1, #model.dec_rnns do
    model.dec_rnns[i]:training()
  end
  for i = 1, #model.dec_atts do
    model.dec_atts[i]:training()
  end

  -- load batch data
  local enc_batch, enc_len_batch, dec_batch = train_loader:random_batch()
  -- ship batch data to gpu
  if opt.gpuid >= 0 then
    enc_batch = enc_batch:float():cuda()
    dec_batch = dec_batch:float():cuda()
  end

  -- do not predict after <E>
  local enc_max_len = enc_batch:size(2)
  local dec_max_len = dec_batch:size(2) - 1

  -- forward propagation ===============================
  if param_x_ ~= param_x then
    param_x:copy(param_x_)
  end

  -- encode
  for i = 1, #model.enc_s[0] do
    model.enc_s[0][i]:zero()
  end
  for i = 1, enc_max_len do
    model.enc_s[i] = model.enc_rnns[i]:forward({enc_batch[{{}, i}], model.enc_s[i - 1]})

    if opt.gpuid >= 0 then
      cutorch.synchronize()
    end
  end
  -- initialize decoder using encoding results
  for i = 1, opt.batch_size do
    for j = 1, #model.dec_s[0] do
      model.dec_s[0][j][{i,{}}]:copy(model.enc_s[enc_len_batch[i]][j][{i,{}}])
    end
  end

  -- build (batch*length*H) for attention score computation
  model.enc_s_top:zero()
  for i = 1, enc_max_len do
    model.enc_s_top[{{},i,{}}]:copy(model.enc_s[i][2*opt.num_layers])
  end

  local enc_s_top_view = model.enc_s_top[{{},{1, enc_max_len},{}}]

  -- decode
  local softmax_predictions = {}
  local loss = 0
  for i = 1, dec_max_len do
    model.dec_s[i] = model.dec_rnns[i]:forward({dec_batch[{{}, i}], model.dec_s[i - 1]})
    softmax_predictions[i] = model.dec_atts[i]:forward({enc_s_top_view, model.dec_s[i][2*opt.num_layers]})
    loss = loss + model.criterions[i]:forward(softmax_predictions[i], dec_batch[{{}, i+1}])

    if opt.gpuid >= 0 then
      cutorch.synchronize()
    end
  end
  loss = loss / opt.batch_size

  -- backward propagation ===============================
  param_dx:zero()
  local enc_ds_top_view = model.enc_ds_top[{{},{1, enc_max_len},{}}]
  enc_ds_top_view:zero()
  reset_ds()
  for i = dec_max_len, 1, -1 do
    local crit_dx = model.criterions[i]:backward(softmax_predictions[i], dec_batch[{{}, i+1}])
    local tmp1, tmp2 = unpack(model.dec_atts[i]:backward({enc_s_top_view, model.dec_s[i][2*opt.num_layers]}, crit_dx))
    enc_ds_top_view:add(tmp1)
    model.ds[2*opt.num_layers]:add(tmp2)
    local tmp = model.dec_rnns[i]:backward({dec_batch[{{}, i}], model.dec_s[i - 1]}, model.ds)[2]
    copy_table(model.ds, tmp)
    if opt.gpuid >= 0 then
      cutorch.synchronize()
    end
  end
  -- back-propagate to encoder
  copy_table(model.save_enc_ds, model.ds)
  local no_blank = false
  for i = enc_max_len, 1, -1 do
    -- mask, or not
    if (not no_blank) then
      no_blank = true
      for j = 1, opt.batch_size do
        if i > enc_len_batch[j] then
          for k = 1, #model.ds do
            model.ds[k][{j,{}}]:zero()
          end
          no_blank = false
        elseif i == enc_len_batch[j] then
          for k = 1, #model.ds do
            model.ds[k][{j,{}}]:copy(model.save_enc_ds[k][{j,{}}])
          end
        end
      end
    end
    -- add gradient from attention layer
    if no_blank then
      model.ds[2*opt.num_layers]:add(model.enc_ds_top[{{},i,{}}])
    else
      for j = 1, opt.batch_size do
        if i <= enc_len_batch[j] then
          model.ds[2*opt.num_layers][{j,{}}]:add(model.enc_ds_top[{j,i,{}}])
        end
      end
    end
    -- bp
    local tmp = model.enc_rnns[i]:backward({enc_batch[{{}, i}], model.enc_s[i - 1]}, model.ds)[2]
    copy_table(model.ds, tmp)
    if opt.gpuid >= 0 then
      cutorch.synchronize()
    end
  end

  -- clip gradient element-wise
  param_dx:clamp(-opt.grad_clip, opt.grad_clip)

  return loss, param_dx
end

function main()
  local cmd = torch.CmdLine()
  cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
  cmd:option('-data_dir', '', 'data path')
  -- bookkeeping
  cmd:option('-seed',123,'torch manual random number generator seed')
  cmd:option('-checkpoint_dir', '', 'output directory where checkpoints get written')
  cmd:option('-savefile','save','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
  cmd:option('-print_every',200,'how many steps/minibatches between printing out the loss')
  -- model params
  cmd:option('-rnn_size', 150, 'size of LSTM internal state')
  cmd:option('-num_layers', 1, 'number of layers in the LSTM')
  cmd:option('-dropout',0.4,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
  cmd:option('-dropoutrec',0,'dropout for regularization, used after each c_i. 0 = no dropout')
  cmd:option('-enc_seq_length',40,'number of timesteps to unroll for')
  cmd:option('-dec_seq_length',100,'number of timesteps to unroll for')
  cmd:option('-batch_size',20,'number of sequences to train on in parallel')
  cmd:option('-max_epochs',95,'number of full passes through the training data')
  -- optimization
  cmd:option('-opt_method', 0, 'optimization method: 0-rmsprop 1-sgd')
  cmd:option('-learning_rate',0.01,'learning rate')
  cmd:option('-init_weight',0.08,'initailization weight')
  cmd:option('-learning_rate_decay',0.985,'learning rate decay')
  cmd:option('-learning_rate_decay_after',5,'in number of epochs, when to start decaying the learning rate')
  cmd:option('-restart',-1,'in number of epochs, when to restart the optimization')
  cmd:option('-decay_rate',0.95,'decay rate for rmsprop')

  cmd:option('-grad_clip',5,'clip gradients at this value')
  cmd:text()
  opt = cmd:parse(arg)

  -- initialize gpu/cpu
  init_device(opt)

  -- setup network
  setup()

  -- load data
  train_loader = MinibatchLoader.create(opt, 'train')

  -- make sure output directory exists
  if not path.exists(opt.checkpoint_dir) then
    lfs.mkdir(opt.checkpoint_dir)
  end

  -- start training
  local step = 0
  local epoch = 0
  local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}

  print("Starting training.")
  local iterations = opt.max_epochs * train_loader.num_batch
  for i = 1, iterations do
    local epoch = i / train_loader.num_batch

    local timer = torch.Timer()
    local loss = 0
    if opt.opt_method == 0 then
      _, loss = optim.rmsprop(eval_training, param_x, optim_state)
    elseif opt.opt_method == 1 then
      _, loss = optim.sgd(eval_training, param_x, optim_state)
    end
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it

    -- exponential learning rate decay
    if opt.opt_method == 0 then
      if i % train_loader.num_batch == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
          local decay_factor = opt.learning_rate_decay
          optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
        end
      end
    end

    if (epoch == opt.restart) and (optim_state.m) then
      optim_state.m:zero()
      optim_state.learningRate = opt.learning_rate
    end

    if i % opt.print_every == 0 then
      print(string.format("%d/%d, train_loss = %6.8f, time/batch = %.2fs", i, iterations, train_loss, time))
    end

    -- on last iteration
    if i == iterations then
      local checkpoint = {}
      checkpoint.enc_rnn_unit = model.enc_rnn_unit
      checkpoint.dec_rnn_unit = model.dec_rnn_unit
      checkpoint.dec_att_unit = model.dec_att_unit
      checkpoint.opt = opt
      checkpoint.i = i
      checkpoint.epoch = epoch

      torch.save(string.format('%s/model.t7', opt.checkpoint_dir), checkpoint)
    end
   
    if i % 30 == 0 then
      collectgarbage()
    end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
      print('loss is NaN.  This usually indicates a bug.')
      break -- halt
    end
  end
end

main()
