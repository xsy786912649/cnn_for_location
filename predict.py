
import tensorflow as tf
import numpy as np

#for a series of inputs(input array), input array shape: (number of samples, 60, 80)
def predict_array(model,input_array):
    input_array=input_array*(-1)
    input_array=np.expand_dims(input_array, 3)
    output=model.predict(input_array)
    psi=output[...,0]/10.0
    epsi=output[...,1]/100.0
    cte=output[...,2]/10.0
    x=output[...,3]*5.0
    y=output[...,4]*5.0
    return psi,epsi,cte,x,y

#for single input, input shape: (60, 80)
def predict_single(model,input):
    input=input*(-1)
    input=np.expand_dims(input, 2)
    output=model.predict(np.array([input]))
    psi=output[0][0]/10.0
    epsi=output[0][1]/100.0
    cte=output[0][2]/10.0
    x=output[0][3]*5.0
    y=output[0][4]*5.0
    return psi,epsi,cte,x,y


#shape of input ()
if __name__ == '__main__':
    saved_model_path='C://Users//wo d pc//Desktop//thelocationi//For Siyuan//mycnnall_2.h5'
    new_model = tf.keras.models.load_model(saved_model_path)
    #new_model.summary()
    pre=predict_single(new_model,input) 
    print(pre)

    