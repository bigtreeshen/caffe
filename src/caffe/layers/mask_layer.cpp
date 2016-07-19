#include <vector>

#include "caffe/layers/mask_layer.hpp"
// #include "caffe/mask_layer.hpp" // for old version caffe

namespace caffe {

template <typename Dtype>
void MaskLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int batch_size = bottom[0]->num();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  top[0]->Reshape(batch_size, 1, height, width); //
}

template <typename Dtype>
void MaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) 
{ 
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_coeff = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  // int offset = 0;
  const int batch_size = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int count = top[0]->count();
  caffe_set(count, Dtype(0), top_data);
  int idx0=0;
  int idx1=0;
  for (int n = 0; n < batch_size; ++n){
  	for (int h = 0; h < height; ++h){
  		for (int w = 0; w < width; ++w){
  			for (int k = 0; k < channels; ++k){
  				idx0=((n * channels + k) * height + h) * width + w;
       				idx1=((n) * height + h) * width + w;
				top_data[idx1] += bottom_coeff[n*channels+k] * bottom_data[idx0];
  			}
  		}
  	}
  }
}

template <typename Dtype>
void MaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) //should have propagate_down
{
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_coeff = bottom[1]->cpu_data();
	Dtype* bottom_data_diff = bottom[0]->mutable_cpu_diff();
	Dtype* bottom_coeff_diff = bottom[1]->mutable_cpu_diff();
	const int batch_size = bottom[0]->num();
	const int channels = bottom[0]->channels();
	const int height = bottom[0]->height();
	const int width = bottom[0]->width();
	int index = 0;
  int top_index=0; //define before using it.
	for (int n = 0; n < batch_size; ++n){
		for (int k = 0; k < channels; ++k){
			Dtype sum = 0;
 			for (int h = 0; h < height; ++h){
				for (int w = 0; w < width; ++w){
					index = ((n * channels + k) * height + h) * width + w;
					top_index = (n * height + h) * width + w;
					sum += top_diff[top_index] * bottom_data[index];
					bottom_data_diff[index] = bottom_coeff[n*channels+k] * top_diff[top_index];
				}
			}
			bottom_coeff_diff[n*channels+k] = sum; //should be inside channel loop
		}
		
	}

}
#ifdef CPU_ONLY
STUB_GPU(MaskLayer);
#endif

INSTANTIATE_CLASS(MaskLayer);
REGISTER_LAYER_CLASS(Mask);

}  // namespace caffe    
