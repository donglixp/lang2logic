require "utils/SymbolsManager.lua"

seq2tree = {}
include 'utils/Tree.lua'

function convert_to_tree(r_list, i_left, i_right, form_manager)
  local t = seq2tree.Tree()
  local level = 0
  local left = -1
  for i = i_left, i_right do
    if r_list[i] == form_manager:get_symbol_idx('(') then
      if level == 0 then
        left = i
      end
      level = level + 1
    elseif r_list[i] == form_manager:get_symbol_idx(')') then
      level = level - 1
      if level == 0 then
        local c = convert_to_tree(r_list, left+1, i-1, form_manager)
        t:add_child(c)
      end
    elseif level == 0 then
      t:add_child(r_list[i])
    end
  end
  return t
end

function process_train_data(opt)
  require('pl.stringx').import()
  require 'pl.seq'

  local timer = torch.Timer()
  
  local data = {}

  local word_manager = SymbolsManager(true)
  word_manager:init_from_file(path.join(opt.data_dir, 'vocab.q.txt'), opt.min_freq, opt.max_vocab_size)
  local form_manager = SymbolsManager(true)
  form_manager:init_from_file(path.join(opt.data_dir, 'vocab.f.txt'), 0, opt.max_vocab_size)

  print('loading text file...')
  local f = torch.DiskFile(path.join(opt.data_dir, opt.train .. '.txt'), 'r', true)
  f:clearError()
  local rawdata = f:readString('*l')
  while (not f:hasError()) do
    local l_list = rawdata:strip():split('\t')
    local w_list = word_manager:get_symbol_idx_for_list(l_list[1]:split(' '))
    local r_list = form_manager:get_symbol_idx_for_list(l_list[2]:split(' '))
    table.insert(data,{w_list,r_list,convert_to_tree(r_list, 1, #r_list, form_manager)})
    -- read next line
    rawdata = f:readString('*l')
  end
  f:close()

  collectgarbage()

  -- save output preprocessed files
  local out_mapfile = path.join(opt.data_dir, 'map.t7')
  print('saving ' .. out_mapfile)
  torch.save(out_mapfile, {word_manager, form_manager})

  collectgarbage()

  local out_datafile = path.join(opt.data_dir, opt.train .. '.t7')
  print('saving ' .. out_datafile)
  torch.save(out_datafile, data)

  collectgarbage()
end

function serialize_data(opt, name)
  require('pl.stringx').import()
  require 'pl.seq'

  local fn = path.join(opt.data_dir, name .. '.txt')

  if not path.exists(fn) then
    print('no file: ' .. fn)
    return nil
  end

  local timer = torch.Timer()
  
  local word_manager, form_manager = unpack(torch.load(path.join(opt.data_dir, 'map.t7')))

  local data = {}

  print('loading text file...')
  local f = torch.DiskFile(fn, 'r', true)
  f:clearError()
  local rawdata = f:readString('*l')
  while (not f:hasError()) do
    local l_list = rawdata:strip():split('\t')
    local w_list = word_manager:get_symbol_idx_for_list(l_list[1]:split(' '))
    local r_list = form_manager:get_symbol_idx_for_list(l_list[2]:split(' '))
    table.insert(data,{w_list,r_list,convert_to_tree(r_list, 1, #r_list, form_manager)})
    -- read next line
    rawdata = f:readString('*l')
  end
  f:close()

  collectgarbage()

  -- save output preprocessed files
  local out_datafile = path.join(opt.data_dir, name .. '.t7')

  print('saving ' .. out_datafile)
  torch.save(out_datafile, data)
end

local cmd = torch.CmdLine()
cmd:option('-data_dir', '', 'data directory')
cmd:option('-train', 'train', 'train data path')
cmd:option('-test', 'test', 'test data path')
cmd:option('-min_freq', 2, 'minimum word frequency')
cmd:option('-max_vocab_size', 15000, 'maximum vocabulary size')
cmd:text()
opt = cmd:parse(arg)

process_train_data(opt)
serialize_data(opt, opt.test)
