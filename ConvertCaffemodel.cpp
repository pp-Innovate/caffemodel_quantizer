#include <iostream>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

#include <boost/shared_ptr.hpp>
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <ristretto/base_ristretto_layer.hpp>

using namespace std;
using namespace boost;
using namespace caffe;

string proto, weight, new_weight;       //file path

void Trim2FixedPoint(float *data, const int cnt, const int bit_width, int fl);
void PrintHelpMessage(void);

int main(int argc, char *argv[])
{
    if(argc == 1)
    {
        PrintHelpMessage();
        return 0;
    }

    /* command line option parse */
    int opt;
    while((opt = getopt(argc, argv, "hm:i:o:")) != -1)
    {
        switch(opt)
        {
            case 'h': PrintHelpMessage();return 0;
            case 'm': proto.assign(optarg);break;       //filename of quantized net definition .prototxt
            case 'i': weight.assign(optarg);break;      //filename of fine-tuned net weight .caffemodel
            case 'o': new_weight.assign(optarg);break;  //filename of quantized net weight .caffemodel
            default: PrintHelpMessage();return 0;
        }
    }

    /* load net definition and weights form file */
    Net<float> network(proto, caffe::TEST);
    network.CopyTrainedLayersFrom(weight);

    /* extract the pointer array of Layer objects */
    vector<shared_ptr<Layer<float> > > ptr_layers = network.layers();

    /* iteration of all layers */
    cout<<"The following layers will be quantized:"<<endl;
    for(unsigned int i = 0; i < ptr_layers.size(); i++)
    {
        /* extract LayerParameter object of current layer */
        LayerParameter lp = ptr_layers[i]->layer_param();

        /* if current layer is not a RistrettoLayer, there is no need to quantize it */
        if(!lp.has_quantization_param())
            continue;

        /* obtain the Ristretto-defined QuantizationParameter */
        int bw_params = lp.quantization_param().bw_params();
        int fl_params = lp.quantization_param().fl_params();

        cout<<" Layer #"<<i<<" : "<<lp.name()<<" : "<<lp.type();
        cout<<"\tbw_params: "<<bw_params<<"\tfl_params: "<<fl_params<<endl;

        /* extract the pointer array of parameter blobs */
        vector<shared_ptr<Blob<float> > > weight_bias = ptr_layers[i]->blobs();

        /* both weights and biased are trimmed to dynamic fixed-point */
        Trim2FixedPoint(weight_bias[0]->mutable_cpu_data(), weight_bias[0]->count(), bw_params, fl_params); 
        Trim2FixedPoint(weight_bias[1]->mutable_cpu_data(), weight_bias[1]->count(), bw_params, fl_params); 
    }

    /* stream quantized net object to a new .caffemodel file */
    NetParameter net_save;
    network.ToProto(&net_save, false);
    WriteProtoToBinaryFile(net_save, new_weight);

    return 0;
}

void Trim2FixedPoint(float *data, const int cnt, const int bit_width, int fl)
{
    float scale = pow(2, -fl);
    float max_data = (pow(2, bit_width - 1) - 1) * scale;
    float min_data = -pow(2, bit_width - 1) * scale;

    for(int i = 0; i < cnt; i++)
    {
        data[i] = std::max(std::min(data[i], max_data), min_data);

        data[i] /= scale;
        data[i] = round(data[i]);
        data[i] *= scale;
    }
}

void PrintHelpMessage(void)
{
    cout<<"Usage: ./ConvertCaffemodel [OPTION]... "<<endl;
    cout<<"Quantize fine-tuned CNN model weights according to the Ristretto-defined QuantizationParameter."<<endl; 
    cout<<"options:"<<endl;
    cout<<" -h\t\tprint this help message."<<endl;
    cout<<" -m\t\tinput quantized model definition .prototxt file."<<endl;
    cout<<" -i\t\tinput fine-tuned model weight .caffemodel file."<<endl;
    cout<<" -o\t\toutput .caffemodel file, value of weights in which are quantized according to \"bw_params\" and \"fl_params\" but type remains 32-bit float."<<endl;
}
