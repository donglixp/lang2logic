local MinibatchLoader = {}
MinibatchLoader.__index = MinibatchLoader

local function to_vector_list(l)
  for i = 1, #l do
    l[i] = l[i][{{},1}]
  end
  return l
end

function MinibatchLoader.create(opt, name)
  local self = {}
  setmetatable(self, MinibatchLoader)

  local data_file = path.join(opt.data_dir, name .. '.t7')

  print('loading data: ' .. name)
  local data = torch.load(data_file)

  -- batch padding
  if #data % opt.batch_size ~= 0 then
    local n = #data
    for i = 1, #data % opt.batch_size do
      table.insert(data, n-i+1, data[n-i+1])
    end
  end
  
  self.enc_batch_list = {}
  self.enc_len_batch_list = {}
  self.dec_batch_list = {}
  local p = 0
  while p + opt.batch_size <= #data do
    -- build enc matrix --------------------------------
    local max_len = #data[p + opt.batch_size][1]
    local m_text = torch.zeros(opt.batch_size, max_len + 2)
    local enc_len_list = {}
    -- add <S>
    m_text[{{}, 1}] = 1
    for i = 1, opt.batch_size do
      local w_list = data[p + i][1]
      for j = 1, #w_list do
        -- reversed order
        m_text[i][j + 1] = w_list[#w_list - j + 1]
        -- m_text[i][j + 1] = w_list[j]
      end
      -- add <E> (for encoder, we need dummy <E> at the end)
      for j = #w_list + 2, max_len +2 do
        m_text[i][j] = 2
      end

      table.insert(enc_len_list, #w_list + 2)
    end
    table.insert(self.enc_batch_list, m_text)
    table.insert(self.enc_len_batch_list, enc_len_list)
    -- build dec matrix --------------------------------
    max_len = -1
    for i = 1, opt.batch_size do
      local w_list = data[p + i][2]
      if #w_list > max_len then
        max_len = #w_list
      end
    end
    m_text = torch.zeros(opt.batch_size, max_len + 2)
    -- add <S>
    m_text[{{}, 1}] = 1
    for i = 1, opt.batch_size do
      local w_list = data[p + i][2]
      for j = 1, #w_list do
        m_text[i][j + 1] = w_list[j]
      end
      -- add <E>
      m_text[i][#w_list + 2] = 2
    end
    table.insert(self.dec_batch_list, m_text)

    p = p + opt.batch_size
  end

  -- reset batch index
  self.num_batch = #self.enc_batch_list

  assert(#self.enc_batch_list == #self.dec_batch_list)

  collectgarbage()
  return self
end

function MinibatchLoader:random_batch()
  local p = math.random(self.num_batch)
  return self.enc_batch_list[p], self.enc_len_batch_list[p], self.dec_batch_list[p]
end

function MinibatchLoader:all_batch()
  local r = {}
  for p = 1, self.num_batch do
    table.insert(r, {self.enc_batch_list[p], self.enc_len_batch_list[p], self.dec_batch_list[p]})
  end
  return r
end

return MinibatchLoader
