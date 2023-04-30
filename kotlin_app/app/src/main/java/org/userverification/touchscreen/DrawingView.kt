package org.userverification.touchscreen

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.widget.TextView
import android.widget.Toast
import androidx.core.content.res.ResourcesCompat
import kotlin.math.abs
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import java.io.File
import java.util.*
import kotlin.collections.ArrayList

private const val STROKE_WIDTH = 12f // has to be float

data class PatternData(
    val size: Int,
    val gyro: ArrayList<FloatArray>,
    val acc: ArrayList<FloatArray>,
    val x: ArrayList<Float>,
    val y: ArrayList<Float>,
    val time: ArrayList<Long>
)

class DrawingView(context: Context, attrs: AttributeSet) : View(context, attrs) {

    private lateinit var extraCanvas: Canvas
    private lateinit var extraBitmap: Bitmap

    private val drawColor = ResourcesCompat.getColor(resources, R.color.black, null)
    private val backgroundColor = ResourcesCompat.getColor(resources, androidx.appcompat.R.color.material_grey_300, null)

    private var hasPreviousDrawing = false

    // Set up the paint with which to draw.
    private val paint = Paint().apply {
        color = drawColor
        // Smooths out edges of what is drawn without affecting shape.
        isAntiAlias = true
        // Dithering affects how colors with higher-precision than the device are down-sampled.
        isDither = true
        style = Paint.Style.STROKE // default: FILL
        strokeJoin = Paint.Join.ROUND // default: MITER
        strokeCap = Paint.Cap.ROUND // default: BUTT
        strokeWidth = STROKE_WIDTH // default: Hairline-width (really thin)
    }

    private var path = Path()

    private var motionTouchEventX = 0f
    private var motionTouchEventY = 0f

    private var currentX = 0f
    private var currentY = 0f

    private val touchTolerance = ViewConfiguration.get(context).scaledTouchSlop

    private val mapper = jacksonObjectMapper()

    var accSensorData: FloatArray? = null // Used for storing newest sensor data
    var gyroSensorData: FloatArray? = null

    private var xDrawingData = ArrayList<Float>()
    private var yDrawingData = ArrayList<Float>()

    private var accList = ArrayList<FloatArray>()
    private var gyroList = ArrayList<FloatArray>()

    private var timeList = ArrayList<Long>()

    var thisViewSize: Int = 0

    override fun onSizeChanged(width: Int, height: Int, oldWidth: Int, oldHeight: Int) {
        super.onSizeChanged(width, height, oldWidth, oldHeight)
        if (::extraBitmap.isInitialized) extraBitmap.recycle()

        extraBitmap = Bitmap.createBitmap(height, width, Bitmap.Config.ARGB_8888)
        extraCanvas = Canvas(extraBitmap)
        extraCanvas.drawColor(backgroundColor)

    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawBitmap(extraBitmap, 0f, 0f, null)
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        motionTouchEventX = event.x
        motionTouchEventY = event.y

        when (event.action) {
            MotionEvent.ACTION_DOWN -> touchStart()
            MotionEvent.ACTION_MOVE -> touchMove()
            MotionEvent.ACTION_UP -> touchUp()
        }
        return true
    }

    private fun touchStart() {
        changeDataText("Drawing: YES", "#006400")
        path.reset()
        path.moveTo(motionTouchEventX, motionTouchEventY)
        currentX = motionTouchEventX
        currentY = motionTouchEventY

        if (hasPreviousDrawing) { // If there is a previous drawing, clear the canvas
            extraCanvas.drawColor(backgroundColor)
        } else {
            hasPreviousDrawing = true
        }

    }

    private fun touchMove() {

        xDrawingData.add(motionTouchEventX)
        yDrawingData.add(motionTouchEventY)

        accList.add(accSensorData!!)
        gyroList.add(gyroSensorData!!)

        timeList.add(System.currentTimeMillis())

        val dx = abs(motionTouchEventX - currentX)
        val dy = abs(motionTouchEventY - currentY)
        if (dx >= touchTolerance || dy >= touchTolerance) {
            // QuadTo() adds a quadratic bezier from the last point,
            // approaching control point (x1,y1), and ending at (x2,y2).
            path.quadTo(currentX, currentY, (motionTouchEventX + currentX) / 2, (motionTouchEventY + currentY) / 2)
            currentX = motionTouchEventX
            currentY = motionTouchEventY
            // Draw the path in the extra bitmap to cache it.
            extraCanvas.drawPath(path, paint)
        }
        invalidate()
    }

    private fun touchUp() {
        // Reset the path so it doesn't get drawn again.
        path.reset()
        changeDataText("Drawing: NO","#880808")

        // Write data
        val pattern = PatternData(
            thisViewSize,
            gyroList,
            accList,
            xDrawingData,
            yDrawingData,
            timeList
        )

        val dateString = Calendar.getInstance().time.toString()
        val currentDirectory = this.context.getExternalFilesDir(null).toString().removeSuffix("files")
        val sampleName = this@DrawingView.rootView.findViewById<TextView>(R.id.textFileName).text.toString()
        if (sampleName != "file name") {
            val saveDir = "$currentDirectory/$sampleName$dateString.json"
            val patternJson = mapper.writeValue(File(saveDir),pattern)
        } else {
            Toast.makeText(this.context, "Brak nazwy pliku!", Toast.LENGTH_SHORT).show()
        }

        // Clear all the data
        xDrawingData.clear()
        yDrawingData.clear()
        accList.clear()
        gyroList.clear()
        timeList.clear()

    }

    private fun changeDataText(text: String, colour: String) {
        this@DrawingView.rootView.findViewById<TextView>(R.id.dataText).text = text
        this@DrawingView.rootView.findViewById<TextView>(R.id.dataText).setTextColor(Color.parseColor(colour))
    }

}