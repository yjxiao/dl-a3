require 'torch'
require 'cunn'
require 'optim'

function train_model(model, criterion, data, labels, valid_data, valid_labels, opt)

    model:cuda()
    criterion:cuda()
    local cudabatch = torch.zeros(opt.minibatchSize, 1, data:size(3), data:size(4)):cuda()
    local cudabatch_labels = torch.zeros(opt.minibatchSize):cuda()
    local best_acc = 0
    parameters, grad_parameters = model:getParameters()
    
    -- optimization functional to train the model with torch's optim library
    local function feval(x) 
        cudabatch[{}] = data:sub(opt.idx, opt.idx+opt.minibatchSize-1)
        cudabatch_labels[{}] = labels:sub(opt.idx, opt.idx+opt.minibatchSize-1)
        
        model:training()
        local minibatch_loss = criterion:forward(model:forward(cudabatch), cudabatch_labels)
        model:zeroGradParameters()
        model:backward(cudabatch, criterion:backward(model.output, cudabatch_labels))
        confusion:batchAdd(model.output, cudabatch_labels)

        return minibatch_loss, grad_parameters
    end
    
    for epoch=1,opt.nEpochs do
        local order = torch.randperm(opt.nBatches) -- not really good randomization
        for batch=1,opt.nBatches do
            --xlua.progress(batch, opt.nBatches)
            opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
            optim.sgd(feval, parameters, opt)
        end

	--print(confusion)
	confusion:updateValids()
	trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
        confusion:zero()

	
	valid_acc = test_model(model, valid_data, valid_labels)
	validLogger:add{['% mean class accuracy (valid set)'] = valid_acc * 100}
	if valid_acc > best_acc then
	    torch.save("/scratch/yx887/dl-a3/results/model_best.net", model)
	    best_acc = valid_acc
	end

    end

end

function test_model(model, data, labels)
    
    model:evaluate()
    local batchSize = 100
    local nBatches = data:size(1) / 100
    local cudabatch = torch.zeros(batchSize, 1, data:size(3), data:size(4)):cuda()
    local cudabatch_labels = torch.zeros(batchSize):cuda()

    confusion:zero()
    for batch=1, nBatches do
        --xlua.progress(batch, nBatches)
        cudabatch[{}] = data:sub((batch-1)*batchSize+1, batch*batchSize)
        cudabatch_labels[{}] = labels:sub((batch-1)*batchSize+1, batch*batchSize)
        local pred = model:forward(cudabatch)
        confusion:batchAdd(pred, cudabatch_labels)
    end
    confusion:updateValids()
    local res = confusion.totalValid
    confusion:zero()

    return res
end

function main()
    -- Configuration parameters
    opt = {}
    opt.dataPath = "/scratch/yx887/dl-a3/data_full_withCap.t7"
    opt.labelPath = "/scratch/yx887/dl-a3/labels_full.t7"

    -- nTrainDocs is the number of documents per class used in the training set, i.e.
    -- here we take the first nTrainDocs documents from each class as training samples
    -- and use the rest as a validation set.
    opt.nTrainDocs = 84500
    opt.nValidDocs = 19500
    opt.nTestDocs = 26000
    opt.nClasses = 5
    -- SGD parameters - play around with these
    opt.nEpochs = 100
    opt.minibatchSize = 100
    opt.nBatches = math.floor(opt.nTrainDocs / opt.minibatchSize)
    opt.learningRate = 3e-2
    opt.learningRateDecay = 3e-5
    opt.momentum = 0.7
    opt.idx = 1

    print("Loading data...")
    local data = torch.load(opt.dataPath)
    local labels = torch.load(opt.labelPath)

    -- split data into makeshift training and validation sets
    local trainSize = opt.nClasses * opt.nTrainDocs
    local testSize = opt.nClasses * opt.nTestDocs
    local validSize = opt.nClasses * opt.nValidDocs
    local training_data = data:sub(1, trainSize)
    local training_labels = labels:sub(1, trainSize)
    local test_data = data:sub(trainSize+1, trainSize+testSize)
    local test_labels = labels:sub(trainSize+1, trainSize+testSize)
    local valid_data = data:sub(trainSize+testSize+1, trainSize+testSize+validSize)
    local valid_labels = labels:sub(trainSize+testSize+1, trainSize+testSize+validSize)

    -- Log results to files
    trainLogger = optim.Logger("/scratch/yx887/dl-a3/results/train.log")
    validLogger = optim.Logger("/scratch/yx887/dl-a3/results/valid.log")
    testLogger = optim.Logger("/scratch/yx887/dl-a3/results/test.log")

    print("Setting up the network...")
    -- initialize confusion matrix
    classes = {'1', '2', '3', '4', '5'}
    confusion = optim.ConfusionMatrix(classes)

    -- construct model:
    ninputs = 1
    nstates = {50, 50, 50}
    noutputs = 5
    
    -- convolution layers
    filtsizeW = {5, 3, 3}       -- filter size across words
    filtsizeH = {51, 1, 1}      -- filter size across word vector dimensions
    poolsizeW = {2, 2, 3}       -- pooling size across words
    poolsizeH = {1, 1, 1}       -- pooling size across word vector dimensions

    model = nn.Sequential()

    model:add(nn.SpatialConvolutionMM(ninputs, nstates[1], filtsizeW[1], filtsizeH[1], 1, filtsizeH[1])) -- 50x196x1
    model:add(nn.ReLU())    
    model:add(nn.SpatialMaxPooling(poolsizeW[1], poolsizeH[1]))		 	           		 -- 50x98x1

    model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsizeW[2], filtsizeH[2], 1, filtsizeH[2]))-- 50x96x1
    model:add(nn.ReLU())    
    model:add(nn.SpatialMaxPooling(poolsizeW[2], poolsizeH[2]))						 -- 50x48x1

    model:add(nn.SpatialConvolutionMM(nstates[2], nstates[3], filtsizeW[3], filtsizeH[3], 1, filtsizeH[3]))-- 50x46x1
    model:add(nn.ReLU())    
    model:add(nn.SpatialMaxPooling(poolsizeW[3], poolsizeH[3]))						 -- 50x15x1

    model:add(nn.Reshape(nstates[3]*15, true))
    model:add(nn.Linear(nstates[3]*15, 5))
    model:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()
    
    print(model)
    print("Training...")   
    train_model(model, criterion, training_data, training_labels, valid_data, valid_labels, opt)
    local results = test_model(model, test_data, test_labels)
    testLogger:add{['% mean class accuracy (test set)'] = results * 100}
end

main()