require 'nn'
require 'cunn'
require 'cudnn'

function createCNNModel(use_cuda)
    local model = nn.Sequential()
    local growthRate = 32
    local nChannels = 2 * growthRate
    
    local function ShareGradInput(module, key)
       assert(key)
       module.__shareGradInputKey = key
       return module
    end

    
    function DenseConnectLayerStandard(nChannels, growthRate)
        local net = nn.Sequential()

        net:add(ShareGradInput(cudnn.SpatialBatchNormalization(nChannels), 'first'))
        net:add(cudnn.ReLU(true))   
        net:add(cudnn.SpatialConvolution(nChannels, 4 * growthRate, 1, 1, 1, 1, 0, 0))
        nChannels = 4 * growthRate
        net:add(cudnn.SpatialBatchNormalization(nChannels))
        net:add(cudnn.ReLU(true))      
        net:add(cudnn.SpatialConvolution(nChannels, growthRate, 3, 3, 1, 1, 1, 1))

        return nn.Sequential()
          :add(nn.Concat(2)
             :add(nn.Identity())
             :add(net))  
    end

    local function addDenseBlock(model, nChannels, N)
          for i = 1, N do 
             model:add(DenseConnectLayerStandard(nChannels, growthRate))
             nChannels = nChannels + growthRate
          end
          return nChannels
       end


    function addTransition(model, nChannels, nOutChannels, last, pool_size)
      model:add(cudnn.SpatialBatchNormalization(nChannels))
      model:add(cudnn.ReLU(true))      
      if last then
         model:add(cudnn.SpatialAveragePooling(pool_size, pool_size))
--          model:add(nn.Reshape(nChannels))      
      else
         model:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0))
         model:add(cudnn.SpatialAveragePooling(2, 2))
      end      
    end
    
    

    -- input shape: (batch_size, 1, imgH, imgW)
    model:add(nn.AddConstant(-128.0))
    model:add(nn.MulConstant(1.0 / 128))
    
    model:add(cudnn.SpatialConvolution(1, nChannels, 3,3, 1,1, 1,1))      

    --Dense-Block 1 and transition
    nChannels = addDenseBlock(model, nChannels, 8)
    addTransition(model, nChannels, math.floor(nChannels*0.5))
    nChannels = math.floor(nChannels*0.5)

--     --Dense-Block 1 and transition
--     nChannels = addDenseBlock(model, nChannels, 15)
--     addTransition(model, nChannels, math.floor(nChannels*0.5))
--     nChannels = math.floor(nChannels*0.5)
    
    --Dense-Block 2 and transition
    nChannels = addDenseBlock(model, nChannels, 11)
    addTransition(model, nChannels, nChannels, true, 4)

    -- (batch_size, 512, H, W)    

    model:add(nn.Transpose({2, 3}, {3,4})) -- (batch_size, H, W, 512)
    model:add(nn.SplitTable(1, 3)) -- #H list of (batch_size, W, 512)
    --model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', false, true))
    --model:cuda()
    return model

end
