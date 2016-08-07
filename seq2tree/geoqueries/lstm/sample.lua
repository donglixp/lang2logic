require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'utils/SymbolsManager'
include "../utils/accuracy.lua"

function transfer_data(x)
  if opt.gpuid >= 0 then
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

function convert_to_string(idx_list)
  local w_list = {}
  for i = 1, #idx_list do
    table.insert(w_list, form_manager:get_idx_symbol(idx_list[i]))
  end
  return table.concat(w_list, ' ')
end

function do_generate(enc_w_list)
  -- encode
  for i = 1, #s do s[i]:zero() end
  -- reversed order
  local enc_w_list_withSE = shallowcopy(enc_w_list)
  table.insert(enc_w_list_withSE,1,word_manager:get_symbol_idx('<E>'))
  table.insert(enc_w_list_withSE,word_manager:get_symbol_idx('<S>'))
  for i = #enc_w_list_withSE, 1, -1 do
    local encoding_result = enc_rnn_unit:forward({transfer_data(torch.Tensor(1):fill(enc_w_list_withSE[i])), s})
    copy_table(s, encoding_result)
  end

  -- decode
  local queue_decode = {}
  table.insert(queue_decode, {s=s, parent=0, child_index=1, t=seq2tree.Tree()})
  local head = 1
  while (head <= #queue_decode) and (head <= 100) do
    s = queue_decode[head].s
    local t = queue_decode[head].t

    local prev_word
    if head == 1 then
      prev_word= float_transfer_data(torch.Tensor(1):fill(form_manager:get_symbol_idx('<S>')))
    else
      prev_word= float_transfer_data(torch.Tensor(1):fill(form_manager:get_symbol_idx('(')))
    end
    local i_child = 1
    while true do
      -- forward the rnn for next word
      local prediction, s_cur = unpack(dec_rnn_unit:forward({prev_word, s}))
      copy_table(s, s_cur)
      
      -- log probabilities from the previous timestep
      local _, _prev_word = prediction:max(2)
      prev_word = _prev_word:resize(1)

      if (prev_word[1] == form_manager:get_symbol_idx('<E>')) or (t.num_children >= checkpoint.opt.dec_seq_length) then
        break
      elseif (prev_word[1] == form_manager:get_symbol_idx('<N>')) then
        table.insert(queue_decode, {s=clone_table(s), parent=head, child_index=i_child, t=seq2tree.Tree()})
        t:add_child(prev_word[1])
      else
        t:add_child(prev_word[1])
      end
      i_child = i_child + 1
    end
    head = head + 1
  end
  -- refine the root tree
  for i = #queue_decode, 2, -1 do
    local cur = queue_decode[i]
    queue_decode[cur.parent].t.children[cur.child_index] = cur.t
  end

  return queue_decode[1].t:to_list(form_manager)
end

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from the learned model')
cmd:text()
cmd:text('Options')
cmd:option('-model','model checkpoint to use for sampling')
cmd:option('-data_dir', '/disk/scratch_ssd/lidong/gen_review/books/', 'data directory')
cmd:option('-input', 'test.t7', 'input data filename')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',0,' 0 to use max at each timestep (-beam_size = 1), 1 to sample at each timestep, 2 to beam search')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-beam_size',20,'beam size')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-display',1,'whether display on console')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
opt.output = opt.model .. '.sample'

-- initialize gpu/cpu
init_device(opt)

-- load the model checkpoint
checkpoint = torch.load(opt.model)
enc_rnn_unit = checkpoint.enc_rnn_unit
dec_rnn_unit = checkpoint.dec_rnn_unit
-- put in eval mode so that dropout works properly
enc_rnn_unit:evaluate()
dec_rnn_unit:evaluate()

-- initialize the rnn state to all zeros
s = {}
local h_init = transfer_data(torch.zeros(1, checkpoint.opt.rnn_size))
for i = 1, checkpoint.opt.num_layers do
  -- c and h for all layers
  table.insert(s, h_init:clone())
  table.insert(s, h_init:clone())
end

-- initialize the vocabulary manager to display text
word_manager, form_manager = unpack(torch.load(path.join(opt.data_dir, 'map.t7')))
-- load data
local data = torch.load(path.join(opt.data_dir, opt.input))

local f_out = torch.DiskFile(opt.output, 'w')
local reference_list = {}
local candidate_list = {}
for i = 1, #data do
  local x = data[i]
  local reference = x[2]
  local candidate = do_generate(x[1])

  table.insert(reference_list, reference)
  table.insert(candidate_list, candidate)

  local ref_str = convert_to_string(reference)
  local cand_str = convert_to_string(candidate)
  -- print to console
  if opt.display > 0 then
    print(ref_str)
    print(cand_str)
    print(' ')
  end
  -- write to file
  f_out:writeString(ref_str)
  f_out:writeString('\n')
  f_out:writeString(cand_str)
  f_out:writeString('\n\n')
   
  if i % 100 == 0 then
    collectgarbage()
  end
end

-- compute evaluation metric
local val_acc = compute_accuracy(candidate_list, reference_list)
print('ACCURACY = ' .. val_acc)
f_out:writeString('ACCURACY = ' .. val_acc)
f_out:writeString('\n')

f_out:close()
