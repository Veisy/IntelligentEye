package com.vyy.intelligenteye.utils


import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.vyy.intelligenteye.ml.EyeModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class ImageClassifierHelper(
    private val context: Context
) {

    var outputAsTensorBuffer: TensorBuffer? = null

    fun classify(image: Bitmap) {
        val eyeModel = EyeModel.newInstance(context)
        val bitmapCopy = image.copy(Bitmap.Config.ARGB_8888, true)

        // Resize bitmap to 224x224
        val resizedBitmap = Bitmap.createScaledBitmap(bitmapCopy, 224, 224, true)

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)

        val byteBuffer = tensorImage.buffer

        // Creates inputs for reference.
        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(byteBuffer)

        // Runs model inference and gets result.
        val outputs = eyeModel.process(inputFeature0)
        outputAsTensorBuffer = outputs.probabilityAsTensorBuffer

        Log.d(
            "Classification Output",
            "classify: ${outputAsTensorBuffer?.floatArray.contentToString()}"
        )

        // Releases model resources if no longer used.
        eyeModel.close()
    }

    fun getOutputMaxIndex(): Int {
        val outputFeature0AsFloatArray = outputAsTensorBuffer?.floatArray
        return outputFeature0AsFloatArray?.indices?.maxByOrNull { outputFeature0AsFloatArray[it] }
            ?: -1
    }

    fun getOutputMaxConfidence(): Float {
        val outputFeature0AsFloatArray = outputAsTensorBuffer?.floatArray
        return outputFeature0AsFloatArray?.maxOrNull() ?: -1f
    }
}