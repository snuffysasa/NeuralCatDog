import sys

picture = sys.argv[1]

def forwardProp(pic = picture):


    
    import numpy as np
    caffe_root = '/home/deep/Desktop/caffe-master/'  # setting the root location to our files
    sys.path.insert(0, caffe_root + 'python')

    import caffe
 
    caffe.set_device(0)
    caffe.set_mode_gpu()
    # using the GPU

    model_def = caffe_root + 'catsdogs/ourNetDefinitions.prototxt'
    model_weights = caffe_root + 'catsdogs/ourTrainedModel.caffemodel'
    #model_weights = caffe_root + 'models/ourModel/catsdogs_train_iter_.caffemodel'

    net = caffe.Net(model_def,      # defines the structure of the model
    model_weights,  # contains the trained weights
    caffe.TEST)     # use test mode 



    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load('python/converted.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    # set the size of the input 
    net.blobs['data'].reshape(1,        # batch size (we are only sending 1 image at a time)
    3,         # 3-channel (BGR) images
    227, 227)  # image size is 227x227


    image = caffe.io.load_image(pic)
    transformed_image = transformer.preprocess('data', image)


    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    ### Do the forward propagation
    output = net.forward()

    output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

    print 'predicted class is:', output_prob.argmax()


    top_inds = output_prob.argsort()[::-1][:2]
    print 'probabilities and labels:', output_prob[top_inds]

forwardProp(picture)

