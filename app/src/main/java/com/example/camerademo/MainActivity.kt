package com.example.camerademo

import android.graphics.*
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlin.math.exp

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val detector = OnnxDetector(this)

        setContent {
            ImageDetectionScreen(detector)
        }
    }
}

@Composable
fun ImageDetectionScreen(detector: OnnxDetector) {
    var originalBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var annotatedBitmap by remember { mutableStateOf<Bitmap?>(null) }

    val context = LocalContext.current
    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            val bmp = BitmapFactory.decodeStream(context.contentResolver.openInputStream(it))
            originalBitmap = bmp

            // Run detection
            val output = detector.run(bmp)
            val detections = parseYoloOutput(output, bmp.width, bmp.height)

            // Draw bounding boxes
            annotatedBitmap = drawBoxesOnBitmap(bmp, detections)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Top
    ) {
        Button(onClick = { launcher.launch("image/*") }) {
            Text("Pick Image")
        }

        Spacer(modifier = Modifier.height(16.dp))

        annotatedBitmap?.let { bmp ->
            Image(
                bitmap = bmp.asImageBitmap(),
                contentDescription = "Detected Image",
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(bmp.width.toFloat() / bmp.height)
            )
        }
    }
}

data class Detection(
    val left: Float,
    val top: Float,
    val width: Float,
    val height: Float,
    val confidence: Float,
    val classId: Int = 0
)

fun parseYoloOutput(
    output: FloatArray,
    bitmapWidth: Int,
    bitmapHeight: Int,
    confThreshold: Float = 0.5f,
    iouThreshold: Float = 0.45f
): List<Detection> {

    Log.d("YOLO_DEBUG", "=== Parsing YOLOv8 Output ===")
    Log.d("YOLO_DEBUG", "Raw output size: ${output.size}")

    // YOLOv8 output format: [1, num_classes+4, num_boxes]
    // For Surfrider dataset - adjust num_classes based on your training
    val numClasses = 10  // UPDATE THIS based on your actual number of classes
    val featuresPerBox = numClasses + 4  // 4 box coords + N class scores
    val numBoxes = output.size / featuresPerBox

    Log.d("YOLO_DEBUG", "Detected shape: $featuresPerBox features x $numBoxes boxes")

    val detections = mutableListOf<Detection>()
    val scaleX = bitmapWidth / 640f
    val scaleY = bitmapHeight / 640f

    // Transpose: read column-wise instead of row-wise
    for (boxIdx in 0 until numBoxes) {
        // Read box coordinates (first 4 values for this box)
        val xCenter = output[boxIdx]
        val yCenter = output[numBoxes + boxIdx]
        val w = output[2 * numBoxes + boxIdx]
        val h = output[3 * numBoxes + boxIdx]

        // Read class scores (remaining 80 values for this box)
        var maxClassScore = 0f
        var maxClassId = 0

        for (classIdx in 0 until numClasses) {
            val score = output[(4 + classIdx) * numBoxes + boxIdx]
            if (score > maxClassScore) {
                maxClassScore = score
                maxClassId = classIdx
            }
        }

        // Confidence is the max class score
        val conf = maxClassScore

        if (conf < confThreshold) continue

        // Convert from 640x640 space to original image space
        val left = (xCenter - w / 2) * scaleX
        val top = (yCenter - h / 2) * scaleY
        val width = w * scaleX
        val height = h * scaleY

        // Clamp to image boundaries
        val clampedLeft = left.coerceIn(0f, bitmapWidth.toFloat())
        val clampedTop = top.coerceIn(0f, bitmapHeight.toFloat())
        val clampedWidth = (left + width).coerceIn(0f, bitmapWidth.toFloat()) - clampedLeft
        val clampedHeight = (top + height).coerceIn(0f, bitmapHeight.toFloat()) - clampedTop

        if (clampedWidth > 0 && clampedHeight > 0) {
            detections.add(
                Detection(
                    left = clampedLeft,
                    top = clampedTop,
                    width = clampedWidth,
                    height = clampedHeight,
                    confidence = conf,
                    classId = maxClassId
                )
            )
        }
    }

    Log.d("YOLO_DEBUG", "Found ${detections.size} detections above threshold $confThreshold")
    detections.take(3).forEach { det ->
        Log.d("YOLO_DEBUG", "Detection: class=${det.classId}, conf=${det.confidence}, box=(${det.left.toInt()},${det.top.toInt()}) ${det.width.toInt()}x${det.height.toInt()}")
    }

    val finalDetections = applyNMS(detections, iouThreshold)
    Log.d("YOLO_DEBUG", "After NMS: ${finalDetections.size} detections")

    return finalDetections
}

fun applyNMS(detections: List<Detection>, iouThreshold: Float): List<Detection> {
    if (detections.isEmpty()) return emptyList()

    // Group by class ID and apply NMS per class
    val byClass = detections.groupBy { it.classId }
    val result = mutableListOf<Detection>()

    byClass.values.forEach { classDetections ->
        val sorted = classDetections.sortedByDescending { it.confidence }
        val keep = mutableListOf<Detection>()
        val suppressed = BooleanArray(sorted.size) { false }

        for (i in sorted.indices) {
            if (suppressed[i]) continue
            keep.add(sorted[i])

            for (j in (i + 1) until sorted.size) {
                if (suppressed[j]) continue
                val iou = calculateIoU(sorted[i], sorted[j])
                if (iou > iouThreshold) {
                    suppressed[j] = true
                }
            }
        }

        result.addAll(keep)
    }

    return result
}

fun calculateIoU(box1: Detection, box2: Detection): Float {
    val x1 = maxOf(box1.left, box2.left)
    val y1 = maxOf(box1.top, box2.top)
    val x2 = minOf(box1.left + box1.width, box2.left + box2.width)
    val y2 = minOf(box1.top + box1.height, box2.top + box2.height)

    val intersectionArea = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
    val box1Area = box1.width * box1.height
    val box2Area = box2.width * box2.height
    val unionArea = box1Area + box2Area - intersectionArea

    return if (unionArea > 0) intersectionArea / unionArea else 0f
}

// Surfrider dataset class names (from data.yaml)
val SURFRIDER_CLASSES = arrayOf(
    "Bottle-shaped",                        // 0
    "Can-shaped",                           // 1
    "Drum",                                 // 2
    "Easily namable",                       // 3
    "Fishing net - cord",                   // 4
    "Insulating material",                  // 5
    "Other packaging",                      // 6
    "Sheet - tarp - plastic bag - fragment",// 7
    "Tire",                                 // 8
    "Unclear"                               // 9
)

fun drawBoxesOnBitmap(bmp: Bitmap, detections: List<Detection>): Bitmap {
    val mutableBmp = bmp.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(mutableBmp)

    val boxPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    val textPaint = Paint().apply {
        color = Color.RED
        textSize = 40f
        style = Paint.Style.FILL
        isFakeBoldText = true
    }

    val bgPaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.FILL
        alpha = 200
    }

    detections.forEach { det ->
        // Draw box
        canvas.drawRect(
            det.left,
            det.top,
            det.left + det.width,
            det.top + det.height,
            boxPaint
        )

        // Draw label with class name and confidence
        val className = if (det.classId < SURFRIDER_CLASSES.size) {
            SURFRIDER_CLASSES[det.classId]
        } else {
            "class_${det.classId}"
        }
        val label = "$className ${String.format("%.0f%%", det.confidence * 100)}"

        val textBounds = Rect()
        textPaint.getTextBounds(label, 0, label.length, textBounds)

        // Background for text
        canvas.drawRect(
            det.left,
            det.top - textBounds.height() - 15f,
            det.left + textBounds.width() + 15f,
            det.top,
            bgPaint
        )

        // Text
        canvas.drawText(label, det.left + 8f, det.top - 8f, textPaint)
    }

    return mutableBmp
}