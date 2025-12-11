package com.example.camerademo

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import android.content.Context
import android.graphics.Bitmap

class OnnxDetector(context: Context) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val options = OrtSession.SessionOptions()
        val modelBytes = context.assets.open("my_model_quantized.onnx").readBytes()
        session = env.createSession(modelBytes, options)
    }

    /**
     * Runs the model on a Bitmap and returns the YOLO output as a flat FloatArray
     */
    fun run(bitmap: Bitmap): FloatArray {
        val input = bitmapToFloatArray(bitmap)

        // ONNX expects [1, 3, 640, 640] input
        val input4d = Array(1) {
            Array(3) { channel ->
                Array(640) { y ->
                    FloatArray(640) { x ->
                        input[channel * 640 * 640 + y * 640 + x]
                    }
                }
            }
        }

        val tensor = OnnxTensor.createTensor(env, input4d)
        val outputList = session.run(mapOf("images" to tensor))
        val rawOutput = outputList[0].value

        // Flatten whatever ONNX returns to 1D FloatArray
        return when (rawOutput) {
            is Array<*> -> flattenArray(rawOutput)
            is FloatArray -> rawOutput
            else -> throw IllegalStateException("Unexpected ONNX output type: ${rawOutput::class}")
        }
    }

    /**
     * Recursively flattens a nested array of Float/FloatArray/Array<*> to 1D FloatArray
     */
    private fun flattenArray(array: Array<*>): FloatArray {
        val list = mutableListOf<Float>()

        fun recurse(a: Any?) {
            when (a) {
                is Float -> list.add(a)
                is FloatArray -> a.forEach { list.add(it) }
                is Array<*> -> a.forEach { recurse(it) }
                else -> throw IllegalStateException("Unknown type in output array: ${a?.javaClass}")
            }
        }

        recurse(array)
        return list.toFloatArray()
    }

    /**
     * Converts a Bitmap to a FloatArray in CHW order and normalized to [0,1]
     */
    private fun bitmapToFloatArray(bmp: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bmp, 640, 640, true)
        val data = FloatArray(3 * 640 * 640)

        val pixels = IntArray(640 * 640)
        resized.getPixels(pixels, 0, 640, 0, 0, 640, 640)

        var offsetR = 0
        var offsetG = 640 * 640
        var offsetB = 2 * 640 * 640

        for (i in pixels.indices) {
            val p = pixels[i]
            data[offsetR++] = ((p shr 16) and 0xFF) / 255f
            data[offsetG++] = ((p shr 8) and 0xFF) / 255f
            data[offsetB++] = (p and 0xFF) / 255f
        }

        return data
    }
}