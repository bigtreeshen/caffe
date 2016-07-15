#include <vector>

#include "caffe/layers/assignment_layer.hpp"
// #include "caffe/assignment_layer.hpp"

namespace caffe {

template <typename Dtype>
void AssignmentLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int batch_size = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  top[0]->Reshape(batch_size, channels, height, width); //
}

template <typename Dtype>
void AssignmentLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) 
{ 
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_mask = bottom[1]->cpu_data();
  const Dtype* bottom_coeff = bottom[2]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int offset = 0;
  const int batch_size = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  int idx = 0;
  int idx_mask = 0;
  for (int n = 0; n < batch_size; ++n){
  	for (int k = 0; k < channels; ++k){
  		for (int h = 0; h < height; ++h){
  			for (int w = 0; w < width; ++w){
  				idx=((n * channels + k) * height + h) * width + w;
  				idx_mask = ((n * height) + h) * width + w;
          			top_data[offset++] = bottom_coeff[(n * channels) + k] * bottom_data[idx] * bottom_mask[idx_mask];
  			}
  		}
  	}
  }

}

template <typename Dtype>
void AssignmentLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) //should have propagate_down
{
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_mask = bottom[1]->cpu_data();
	const Dtype* bottom_coeff = bottom[2]->cpu_data();
	Dtype* bottom_data_diff = bottom[0]->mutable_cpu_diff();
	Dtype* bottom_mask_diff = bottom[1]->mutable_cpu_diff();
	Dtype* bottom_coeff_diff = bottom[2]->mutable_cpu_diff();
	const int batch_size = bottom[0]->num();
	const int channels = bottom[0]->channels();
	const int height = bottom[0]->height();
	const int width = bottom[0]->width();
	int index = 0;
	int mask_index=0;
	const int mask_count=bottom[1]->count();
	caffe_set(mask_count,Dtype(0),bottom_mask_diff);
	for (int n = 0; n < batch_size; ++n){
		for (int k = 0; k < channels; ++k){
			Dtype sum = 0;
			for (int h = 0; h < height; ++h){
				for (int w = 0; w < width; ++w){
					index = ((n * channels + k) * height + h) * width + w;
					mask_index = (n * height + h) * width + w;
					sum += top_diff[index] * bottom_data[index] * bottom_mask[mask_index];
					bottom_data_diff[index] = bottom_coeff[n * channels + k] * top_diff[index] * bottom_mask[mask_index];
					bottom_mask_diff[mask_index] += top_diff[index] * bottom_coeff[(n * channels) + k] * bottom_data[index];
				}
			}
			bottom_coeff_diff[n*channels + k] = sum; //should be inside channel loop
		}
	}


}
#ifdef CPU_ONLY
STUB_GPU(MaskLayer);
#endif

INSTANTIATE_CLASS(AssignmentLayer);
REGISTER_LAYER_CLASS(Assignment);

}  // namespace caffe    
