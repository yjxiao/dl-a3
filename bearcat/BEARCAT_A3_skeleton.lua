-------------------------------------------------------------------------
-- In this part of the assignment you will become more familiar with the
-- internal structure of torch modules and the torch documentation.
-- You must complete the definitions of updateOutput and updateGradInput
-- for a 1-d log-exponential pooling module as explained in the handout.
-- 
-- Refer to the torch.nn documentation of nn.TemporalMaxPooling for an
-- explanation of parameters kW and dW.
-- 
-- Refer to the torch.nn documentation overview for explanations of the 
-- structure of nn.Modules and what should be returned in self.output 
-- and self.gradInput.
-- 
-- Don't worry about trying to write code that runs on the GPU.
--
-- Please find submission instructions on the handout
------------------------------------------------------------------------

local TemporalLogExpPooling, parent = torch.class('nn.TemporalLogExpPooling', 'nn.Module')

function TemporalLogExpPooling:__init(kW, dW, beta)
   parent.__init(self)

   self.kW = kW
   self.dW = dW
   self.beta = beta

   self.indices = torch.Tensor()
end

function TemporalLogExpPooling:updateOutput(input)
   -----------------------------------------------
   -- your code here
   --
   -- We expect the input to be 2 or 3 dimension
   -- where the first dimension (optional) is
   -- sample size, the second and third dimensions
   -- are input size and number of channels
   -- respectively.
   -----------------------------------------------
   
   local size = input:size()
   if size:size() == 2 then
      input = torch.repeatTensor(input, 1, 1, 1)
      n = 1         -- number of samples
      d = size[1]   -- input dimension
      k = size[2]   -- number of channels (FrameSize)
   elseif size:size() == 3 then
      n = size[1]
      d = size[2]
      k = size[3]
   else
      print 'input dimension 2 or 3 expected!'
   end

   local d_output = torch.floor((d - self.kW) / self.dW + 1)
   self.output = torch.zeros(n, d_output, k)
   for i = 1, d_output do
      self.output[{{}, {i}, {}}] = torch.mul(input[{{}, {(i-1)*self.dW+1, (i-1)*self.dW+self.kW}, {}}], self.beta):exp():sum(2):div(self.kW):log():div(self.beta)
   end
      
   return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)
   -----------------------------------------------
   -- your code here
   -----------------------------------------------

   local size = gradOutput:size()
   local n = size[1]
   local d = size[2]
   local k = size[3]
   
   self.gradInput = torch.zeros(input:size())
   for i = 1, d do
      local exp_input = torch.mul(input[{{}, {(i-1)*self.dW+1, (i-1)*self.dW+self.kW}, {}}], self.beta):exp()
      local sum_exp = exp_input:sum(2)
      self.gradInput[{{}, {(i-1)*self.dW+1, (i-1)*self.dW+self.kW}, {}}]:add(exp_input:cdiv(sum_exp:expand(n, self.kW, k)))
   end
   
   return self.gradInput
end

function TemporalLogExpPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end
