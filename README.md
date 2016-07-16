This my own Caffe branch, work with ZHAN Huangying.

Two layers added by us in this branch, which are,

#MaskLayer
Using feature map and correspond coefficients to generate mask.
##usage

layer  
{   
        name: "mask_layer"  
        type: "Mask"  
        bottom: "pool5"  
        bottom: "fc10_alpha"  
        top: "mask_output"  
}

#AssignmentLayer
Using mask and feature map to generate new features.
##usage

layer  
{  
        name: "assignment_layer"  
        type: "Assignment"  
        bottom: "pool5"  
        bottom: "mask_output_normed"  
        bottom: "fc10_beta"  
        top: "pool5_1"  
}
