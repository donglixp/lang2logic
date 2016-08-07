require 'torch'
require 'nn'
require 'nngraph'
local class = require 'class'
require 'optim'
require('pl.stringx').import()
require 'pl.seq'
require 'utils/SymbolsManager'
include "../utils/utils.lua"
include '../utils/MinibatchLoader.lua'
include '../utils/ResourceManager.lua'

function transfer_data(x)
  if opt.gpuid>=0 then
    return x:cuda()
  end
  return x
end

function float_transfer_data(x)
  if opt.gpuid>=0 then
    return x:float():cuda()
  end
  return x
end

function lstm_enc(x, prev_c, prev_h)
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

function lstm_dec(x, prev_c, prev_h, parent_h)
  -- Calculate all four gates in one go
  local i = nn.JoinTable(2)({x, parent_h})
  local i2h = nn.Linear(2 * opt.rnn_size, 4 * opt.rnn_size)(i)
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

function create_enc_lstm_unit(w_size)
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
    local next_c, next_h = lstm_enc(x_in, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local m = nn.gModule({x, prev_s}, {nn.Identity()(next_s)})
  
  return transfer_data(m)
end

function create_dec_lstm_unit(w_size)
  -- input
  local x = nn.Identity()()
  local prev_s = nn.Identity()()
  local parent_h = nn.Identity()()

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
    local next_c, next_h = lstm_dec(x_in, prev_c, prev_h, parent_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local m = nn.gModule({x, prev_s, parent_h}, {nn.Identity()(next_s)})
  
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
  
  for i = 0, opt.dec_seq_length do
    model.dec_s[i] = {}
    for j = 0, opt.dec_seq_length do
      model.dec_s[i][j] = {}
      for d = 1, 2 * opt.num_layers do
        model.dec_s[i][j][d] = {}
      end
    end
  end

  for d = 1, 2 * opt.num_layers do
    model.ds[d] = transfer_data(torch.zeros(opt.batch_size, opt.rnn_size))
    model.save_enc_ds[d] = transfer_data(torch.zeros(opt.batch_size, opt.rnn_size))
  end

  word_manager, form_manager = unpack(torch.load(path.join(opt.data_dir, 'map.t7')))

  print("Creating encoder")
  model.enc_rnn_unit = create_enc_lstm_unit(word_manager.vocab_size)

  print("Creating decoder")
  model.dec_rnn_unit = create_dec_lstm_unit(form_manager.vocab_size)
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

function save_parent_h_ds_to_parent(i_batch, par_index, child_index, parent_ds_save_table, parent_h_ds)
  if parent_ds_save_table[i_batch] == nil then parent_ds_save_table[i_batch] = {} end
  if parent_ds_save_table[i_batch][par_index] == nil then parent_ds_save_table[i_batch][par_index] = {} end
  if parent_ds_save_table[i_batch][par_index][child_index] == nil then
    parent_ds_save_table[i_batch][par_index][child_index] = {}
    for d = 1, 2*opt.num_layers-1 do
      parent_ds_save_table[i_batch][par_index][child_index][d] = transfer_data(torch.zeros(opt.batch_size, opt.rnn_size))
    end
    parent_ds_save_table[i_batch][par_index][child_index][2*opt.num_layers] = parent_h_ds:clone()
  else
    parent_ds_save_table[i_batch][par_index][child_index][2*opt.num_layers]:add(parent_h_ds)
  end
end

function save_ds_to_parent(i_batch, par_index, child_index, parent_ds_save_table)
  if parent_ds_save_table[i_batch] == nil then parent_ds_save_table[i_batch] = {} end
  if parent_ds_save_table[i_batch][par_index] == nil then parent_ds_save_table[i_batch][par_index] = {} end
  if parent_ds_save_table[i_batch][par_index][child_index] == nil then
    parent_ds_save_table[i_batch][par_index][child_index] = clone_table(model.ds)
  else
    add_table(parent_ds_save_table[i_batch][par_index][child_index], model.ds)
  end
end

function add_ds_to_parent(cur_index, child_index, parent_ds_save_table)
  for i_batch = 1, opt.batch_size do
    if (parent_ds_save_table[i_batch] ~= nil) and (parent_ds_save_table[i_batch][cur_index] ~= nil) and (parent_ds_save_table[i_batch][cur_index][child_index] ~= nil) then
      -- add the i_batch gradient vector to parent
      local child_ds = parent_ds_save_table[i_batch][cur_index][child_index]
      for d = 1, 2*opt.num_layers do
        model.ds[d][{i_batch,{}}]:add(child_ds[d][{i_batch,{}}])
      end
    end
  end
end

function eval_training(param_x_)
  model.enc_rnn_unit:training()
  model.dec_rnn_unit:training()
  model.dec_att_unit:training()
  for i = 1, #model.enc_rnns do model.enc_rnns[i]:training() end
  for i = 1, #model.dec_rnns do model.dec_rnns[i]:training() end
  for i = 1, #model.dec_atts do model.dec_atts[i]:training() end

  local dec_rnns_manager = seq2tree.ResourceManager()
  dec_rnns_manager:reset(#model.dec_rnns)
  
  -- load batch data
  local enc_batch, enc_len_batch, dec_tree_batch = train_loader:random_batch()
  -- ship batch data to gpu
  enc_batch = float_transfer_data(enc_batch)

  local enc_max_len = enc_batch:size(2)

  -- forward propagation ===============================
  if param_x_ ~= param_x then
    param_x:copy(param_x_)
  end

  -- encode
  for i = 1, #model.enc_s[0] do
    model.enc_s[0][i]:zero()
  end
  for i = 1, enc_max_len do
    local tmp = model.enc_rnns[i]:forward({enc_batch[{{}, i}], model.enc_s[i - 1]})
    copy_table(model.enc_s[i], tmp)
  end

  -- build (batch*length*H) for attention score computation
  model.enc_s_top:zero()
  for i = 1, enc_max_len do
    model.enc_s_top[{{},i,{}}]:copy(model.enc_s[i][2*opt.num_layers])
  end

  local enc_s_top_view = model.enc_s_top[{{},{1, enc_max_len},{}}]
  
  -- decode
  local queue_tree = {}
  for i = 1, opt.batch_size do
    queue_tree[i] = {}
    table.insert(queue_tree[i], {tree=dec_tree_batch[i], parent=0, child_index=1})
  end
  local softmax_predictions = {}
  local loss = 0
  local cur_index, max_index = 1, 1
  local dec_batch = {}
  while (cur_index <= max_index) do
    -- build dec_batch for cur_index
    local max_w_len=-1
    local batch_w_list = {}
    for i = 1, opt.batch_size do
      local w_list = {}
      if (cur_index <= #queue_tree[i]) then
        local t = queue_tree[i][cur_index].tree
        for ic = 1, t.num_children do
          if class.istype(t.children[ic], 'seq2tree.Tree') then
            -- non-terminal symbol (4)
            table.insert(w_list, 4)
            table.insert(queue_tree[i], {tree=t.children[ic], parent=cur_index, child_index=ic})
          else
            table.insert(w_list, t.children[ic])
          end
        end
        if #queue_tree[i] > max_index then max_index = #queue_tree[i] end
      end
      if #w_list>max_w_len then max_w_len = #w_list end
      table.insert(batch_w_list, w_list)
    end
    dec_batch[cur_index] = torch.zeros(opt.batch_size, max_w_len + 2)
    for i = 1, opt.batch_size do
      local w_list = batch_w_list[i]
      if #w_list > 0 then
        for j = 1, #w_list do dec_batch[cur_index][i][j + 1] = w_list[j] end
        -- add <S>, <E>
        if cur_index == 1 then
          dec_batch[cur_index][i][1] = 1
        else
          dec_batch[cur_index][i][1] = form_manager:get_symbol_idx('(')
        end
        dec_batch[cur_index][i][#w_list + 2] = 2
      end
    end
    dec_batch[cur_index] = float_transfer_data(dec_batch[cur_index])
    
    -- initialize first decoder unit hidden state
    for j = 1, 2 * opt.num_layers do
      model.dec_s[cur_index][0][j] = transfer_data(torch.zeros(opt.batch_size, opt.rnn_size))
    end
    if (cur_index==1) then
      -- initialize using encoding results
      for i = 1, opt.batch_size do
        for j = 1, 2 * opt.num_layers do
          model.dec_s[1][0][j][{i,{}}]:copy(model.enc_s[enc_len_batch[i]][j][{i,{}}])
        end
      end
    else
      for i = 1, opt.batch_size do
        if (cur_index <= #queue_tree[i]) then
          local par_index = queue_tree[i][cur_index].parent
          local child_index = queue_tree[i][cur_index].child_index
          for j = 1, 2 * opt.num_layers do
            model.dec_s[cur_index][0][j][{i,{}}]:copy(model.dec_s[par_index][child_index][j][{i,{}}])
          end
        end
      end
    end

    softmax_predictions[cur_index] = {}
    local parent_h = model.dec_s[cur_index][0][2*opt.num_layers]
    -- do not predict after <E>
    for i = 1, dec_batch[cur_index]:size(2) - 1 do
      local i_dec_rnns = dec_rnns_manager:allocate2(cur_index, i)
      model.dec_s[cur_index][i] = model.dec_rnns[i_dec_rnns]:forward({dec_batch[cur_index][{{}, i}], model.dec_s[cur_index][i - 1], parent_h})
      softmax_predictions[cur_index][i] = model.dec_atts[i_dec_rnns]:forward({enc_s_top_view, model.dec_s[cur_index][i][2*opt.num_layers]})
      loss = loss + model.criterions[i_dec_rnns]:forward(softmax_predictions[cur_index][i], dec_batch[cur_index][{{}, i+1}])
    end

    cur_index = cur_index + 1
  end
  loss = loss / opt.batch_size

  -- backward propagation ===============================
  param_dx:zero()
  model.enc_ds_top:zero()
  local enc_ds_top_view = model.enc_ds_top[{{},{1, enc_max_len},{}}]

  local parent_ds_save_table = {}
  local save_enc_parent_h_ds = transfer_data(torch.zeros(opt.batch_size, opt.rnn_size))
  for cur_index = max_index, 1, -1 do
    reset_ds()
    local parent_h = model.dec_s[cur_index][0][2*opt.num_layers]
    for i = dec_batch[cur_index]:size(2) - 1, 1, -1 do
      local i_dec_rnns = dec_rnns_manager:get2(cur_index, i)
      add_ds_to_parent(cur_index, i, parent_ds_save_table)
      local crit_dx = model.criterions[i_dec_rnns]:backward(softmax_predictions[cur_index][i], dec_batch[cur_index][{{}, i+1}])

      local tmp1, tmp2 = unpack(model.dec_atts[i_dec_rnns]:backward({enc_s_top_view, model.dec_s[cur_index][i][2*opt.num_layers]}, crit_dx))
      enc_ds_top_view:add(tmp1)
      model.ds[2*opt.num_layers]:add(tmp2)
      
      local _, tmp, parent_h_ds = unpack(model.dec_rnns[i_dec_rnns]:backward({dec_batch[cur_index][{{}, i}], model.dec_s[cur_index][i - 1], parent_h}, model.ds))
      copy_table(model.ds, tmp)
      -- bp the parent_h_ds to the parent node
      if (cur_index > 1) then
        for i_batch = 1, opt.batch_size do
          if (cur_index <= #queue_tree[i_batch]) then
            local par_index = queue_tree[i_batch][cur_index].parent
            local child_index = queue_tree[i_batch][cur_index].child_index
            save_parent_h_ds_to_parent(i_batch, par_index, child_index, parent_ds_save_table, parent_h_ds)
          end
        end
      elseif (cur_index == 1) then
        save_enc_parent_h_ds:add(parent_h_ds)
      end
    end
    -- save and add to the parent node
    if (cur_index > 1) then
      for i_batch = 1, opt.batch_size do
        if (cur_index <= #queue_tree[i_batch]) then
          local par_index = queue_tree[i_batch][cur_index].parent
          local child_index = queue_tree[i_batch][cur_index].child_index
          save_ds_to_parent(i_batch, par_index, child_index, parent_ds_save_table)
        end
      end
    end
  end
  model.ds[2*opt.num_layers]:add(save_enc_parent_h_ds)

  -- back-propagate to encoder
  copy_table(model.save_enc_ds, model.ds)
  local no_blank = false
  for i = enc_max_len, 1, -1 do
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
  cmd:option('-print_every',2000,'how many steps/minibatches between printing out the loss')
  -- model params
  cmd:option('-rnn_size', 200, 'size of LSTM internal state')
  cmd:option('-num_layers', 1, 'number of layers in the LSTM')
  cmd:option('-dropout',0.3,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
  cmd:option('-dropoutrec',0.3,'dropout for regularization, used after each c_i. 0 = no dropout')
  cmd:option('-enc_seq_length',60,'number of timesteps to unroll for')
  cmd:option('-dec_seq_length',220,'number of timesteps to unroll for')
  cmd:option('-batch_size',20,'number of sequences to train on in parallel')
  cmd:option('-max_epochs',130,'number of full passes through the training data')
  -- optimization
  cmd:option('-opt_method', 0, 'optimization method: 0-rmsprop 1-sgd')
  cmd:option('-learning_rate',0.007,'learning rate')
  cmd:option('-init_weight',0.08,'initailization weight')
  cmd:option('-learning_rate_decay',0.98,'learning rate decay')
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
  train_loader = seq2tree.MinibatchLoader()
  train_loader:create(opt, 'train')

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
  local loss0 = nil
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
    if (opt.opt_method == 0) then
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
