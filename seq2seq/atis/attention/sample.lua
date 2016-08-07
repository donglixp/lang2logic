require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'utils/SymbolsManager'
include "../utils/utils.lua"
include "../utils/accuracy.lua"

function transfer_data(x)
  if opt.gpuid >= 0 then
    return x:cuda()
  end
  return x
end

function convert_to_string(idx_list, f_out)
  local w_list = {}
  for i = 1, #idx_list do
    table.insert(w_list, form_manager:get_idx_symbol(idx_list[i]))
  end
  return table.concat(w_list, ' ')
end

function generate_next_word(hs, prev_word_idx, enc_s_top)
  local prev_word
  if opt.gpuid >= 0 then
    prev_word = (torch.Tensor(1):fill(prev_word_idx)):float():cuda()
  else
    prev_word = torch.Tensor(1):fill(prev_word_idx)
  end
  -- forward the rnn for next word
  local s_cur = dec_rnn_unit:forward({prev_word, hs})
  local prediction = dec_att_unit:forward({enc_s_top, s_cur[2*checkpoint.opt.num_layers]})

  local val, w_idx = prediction:sort(2, true)

  return w_idx:view(w_idx:nElement()):narrow(1,1,opt.beam_size):clone(),
    val:view(val:nElement()):narrow(1,1,opt.beam_size):clone(), clone_table(s_cur)
end

function do_generate(enc_w_list)
  -- encode
  for i = 1, #s do
    s[i]:zero()
  end
  -- reversed order
  local enc_w_list_withSE = shallowcopy(enc_w_list)
  table.insert(enc_w_list_withSE,1,word_manager:get_symbol_idx('<E>'))
  table.insert(enc_w_list_withSE,word_manager:get_symbol_idx('<S>'))
  local enc_s_top = transfer_data(torch.zeros(1, #enc_w_list_withSE, checkpoint.opt.rnn_size))
  for i = #enc_w_list_withSE, 1, -1 do
    local encoding_result = enc_rnn_unit:forward({transfer_data(torch.Tensor(1):fill(enc_w_list_withSE[i])), s})
    copy_table(s, encoding_result)

    enc_s_top[{{}, #enc_w_list_withSE-i+1, {}}]:copy(s[2*checkpoint.opt.num_layers])
  end

  -- decode
  if opt.sample == 0 or opt.sample == 1 then
    local text_gen = {}

    local prev_word
    if opt.gpuid >= 0 then
      prev_word = (torch.Tensor(1):fill(form_manager:get_symbol_idx('<S>'))):float():cuda()
    else
      prev_word = torch.Tensor(1):fill(form_manager:get_symbol_idx('<S>'))
    end

    while true do
      -- forward the rnn for next word
      local s_cur = dec_rnn_unit:forward({prev_word, s})
      local prediction = dec_att_unit:forward({enc_s_top, s_cur[2*checkpoint.opt.num_layers]})

      copy_table(s, s_cur)
      
      -- log probabilities from the previous timestep
      if opt.sample == 0 then
        -- use argmax
        local _, _prev_word = prediction:max(2)
        prev_word = _prev_word:resize(1)
      else
        -- use sampling
        prediction:div(opt.temperature) -- scale by temperature
        local probs = torch.exp(prediction):squeeze()
        probs:div(torch.sum(probs)) -- renormalize so probs sum to one
        prev_word = torch.multinomial(probs:float(), 1):resize(1):float()
      end

      if (prev_word[1] == form_manager:get_symbol_idx('<E>')) or (#text_gen >= checkpoint.opt.dec_seq_length) then
        break
      else
        table.insert(text_gen, prev_word[1])
      end
    end

    return text_gen
  else
    local beam_list = {{prb = 0, text_gen = {}, s = s}}
    while true do
      local search_list = {}
      for i = 1, #beam_list do
        local h = beam_list[i]
        local last_word
        if #h.text_gen == 0 then
          last_word = form_manager:get_symbol_idx('<S>')
        else
          last_word = h.text_gen[#h.text_gen]
        end
        if last_word == form_manager:get_symbol_idx('<E>') then
          table.insert(search_list, h)
        else
          local w_new_list, p_list, s_cur = generate_next_word(h.s, last_word, enc_s_top)
          for j = 1, w_new_list:nElement() do
            local w_new = w_new_list[j]
            local p = p_list[j]
            local text_gen_append = shallowcopy(h.text_gen)
            table.insert(text_gen_append, w_new)
            table.insert(search_list, {prb = h.prb + p, text_gen = text_gen_append, s = s_cur})
          end
        end
      end
      -- sort and get the new beam list
      table.sort(search_list, function(a,b) return a.prb > b.prb end)
      beam_list = table_topk(search_list, opt.beam_size)
      -- whether stop generating
      local is_all_end = true
      for i = 1, #beam_list do
        local h = beam_list[i]
        local last_word = h.text_gen[#h.text_gen]
        if last_word ~= form_manager:get_symbol_idx('<E>') and #h.text_gen < checkpoint.opt.dec_seq_length then
          is_all_end = false
          break
        end
      end
      if is_all_end then
        break
      end
    end
    -- return the first one (max probability)
    local text_gen = beam_list[1].text_gen
    -- remove the last <E>
    if text_gen[#text_gen] == form_manager:get_symbol_idx('<E>') then
      table.remove(text_gen)
    end
    return text_gen
  end
end

function post_process(candidate)
  local c_left = 0
  local c_right = 0
  for i = 1, #candidate do
    if (candidate[i] == form_manager:get_symbol_idx('(')) then
      c_left = c_left +1
    elseif (candidate[i] == form_manager:get_symbol_idx(')')) then
      c_right = c_right +1
    end
  end

  if c_right > c_left then
    for j = 1, c_right - c_left do
      table.remove(candidate)
    end
  end
  if c_right < c_left then
    for j = 1, c_left - c_right do
      table.insert(candidate, form_manager:get_symbol_idx(')'))
    end
  end

  return candidate
end

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from the learned model')
cmd:text()
cmd:text('Options')
cmd:option('-model','model checkpoint to use for sampling')
cmd:option('-data_dir', '', 'data directory')
cmd:option('-input', 'test.t7', 'input data filename')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',0,' 0 to use max at each timestep (-beam_size = 1), 1 to sample at each timestep, 2 to beam search')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-beam_size',5,'beam size')
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
dec_att_unit = checkpoint.dec_att_unit
-- put in eval mode so that dropout works properly
enc_rnn_unit:evaluate()
dec_rnn_unit:evaluate()
dec_att_unit:evaluate()

-- initialize the rnn state to all zeros
s = {}
for i = 1, 2*checkpoint.opt.num_layers do
  -- c and h for all layers
  table.insert(s, transfer_data(torch.zeros(1, checkpoint.opt.rnn_size)))
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

  candidate = post_process(candidate)

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
local val_acc = compute_tree_accuracy(candidate_list, reference_list, form_manager)
print('ACCURACY = ' .. val_acc)
f_out:writeString('ACCURACY = ' .. val_acc)
f_out:writeString('\n')

f_out:close()
