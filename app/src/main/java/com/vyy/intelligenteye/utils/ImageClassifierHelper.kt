package com.vyy.intelligenteye.utils


import android.content.Context
import android.content.res.Resources
import android.graphics.Bitmap
import android.util.Log
import com.vyy.intelligenteye.ml.EyeModel
import com.vyy.intelligenteye.processes.resize
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class ImageClassifierHelper(
    private val context: Context
) {


    fun classify(image: Bitmap, resources: Resources) {
        val model = EyeModel.newInstance(context)

        val bitmapCopy = image.copy(Bitmap.Config.ARGB_8888, true)

        val bitmap = resize(bitmapCopy, 224, 224, resources).bitmap

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)

        val byteBuffer = tensorImage.buffer

        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(byteBuffer)

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        Log.d("RESULT", outputFeature0.floatArray.contentToString())

        // Releases model resources if no longer used.
        model.close()
    }

    companion object {
        const val MAX_RESULTS = 1

        private const val TAG = "ImageClassifierHelper"
    }
}