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
import android.widget.ToggleButton
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
    val gProperties: MutableList<SensorProperties>,
    val aProperties: MutableList<SensorProperties>,
    val x: ArrayList<Float>,
    val y: ArrayList<Float>,
    val time: ArrayList<Long>,
    val rawTime: ArrayList<Long>,
    val toggleButton1Info: String,
    val toggleButton2Info: String,
    val toggleButton3Info: String
)

data class SensorProperties(
    val resolution: Float,
    val maxRange: Float,
    val minDelay: Int,
    val maxDelay: Int,
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

    var accSensorData: FloatArray = floatArrayOf(0.0f, 0.0f, 0.0f) // Used for storing newest sensor data
    var gyroSensorData: FloatArray = floatArrayOf(0.0f, 0.0f, 0.0f)

    var accProperties = mutableListOf<SensorProperties>()
    var gyroProperties = mutableListOf<SensorProperties>()

    private var xDrawingData = ArrayList<Float>()
    private var yDrawingData = ArrayList<Float>()

    private var accList = ArrayList<FloatArray>()
    private var gyroList = ArrayList<FloatArray>()

    private var timeFirst = 0L
    private var timeSecond = 0L

    private var elapsedTimeList = ArrayList<Long>()
    private var rawTimeList = ArrayList<Long>()

    private var toggleInfo1 = String()
    private var toggleInfo2 = String()
    private var toggleInfo3 = String()

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
        timeFirst = System.nanoTime()
        path.reset()
        path.moveTo(motionTouchEventX, motionTouchEventY)
        currentX = motionTouchEventX
        currentY = motionTouchEventY

        addData()
        elapsedTimeList.add(0L)
        rawTimeList.add(System.nanoTime())

        if (hasPreviousDrawing) { // If there is a previous drawing, clear the canvas
            extraCanvas.drawColor(backgroundColor)
        } else {
            hasPreviousDrawing = true
        }

    }

    private fun touchMove() {

        addData()

        timeSecond = System.nanoTime()
        elapsedTimeList.add(timeSecond - timeFirst)
        timeFirst = System.nanoTime()
        rawTimeList.add(timeFirst)

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

        val toggleButton1  = this@DrawingView.rootView.findViewById<ToggleButton>(R.id.toggleButton)
        val toggleButton2  = this@DrawingView.rootView.findViewById<ToggleButton>(R.id.toggleButton2)
        val toggleButton3  = this@DrawingView.rootView.findViewById<ToggleButton>(R.id.toggleButton3)

        toggleInfo1 =  if (toggleButton1.isChecked) {
            toggleButton1.textOn.toString()
        } else {
            toggleButton1.textOff.toString()
        }

        toggleInfo2 =  if (toggleButton2.isChecked) {
            toggleButton2.textOn.toString()
        } else {
            toggleButton2.textOff.toString()
        }

        toggleInfo3 =  if (toggleButton3.isChecked) {
            toggleButton3.textOn.toString()
        } else {
            toggleButton3.textOff.toString()
        }

        // Write data
        val pattern = PatternData(
            thisViewSize,
            gyroList,
            accList,
            gyroProperties,
            accProperties,
            xDrawingData,
            yDrawingData,
            elapsedTimeList,
            rawTimeList,
            toggleInfo1,
            toggleInfo2,
            toggleInfo3
        )

        val dateString = Calendar.getInstance().time.toString()
        val currentDirectory = this.context.getExternalFilesDir(null).toString().removeSuffix("files")
        val sampleName = this@DrawingView.rootView.findViewById<TextView>(R.id.textFileName).text.toString()
        if (sampleName != "file name") {
            val saveDir = "$currentDirectory/$sampleName$dateString.json"
            val patternJson = mapper.writeValue(File(saveDir),pattern)
            println("saved file")
        } else {
            Toast.makeText(this.context, "Brak nazwy pliku!", Toast.LENGTH_SHORT).show()
        }

        // Clear all the data
        xDrawingData.clear()
        yDrawingData.clear()
        accList.clear()
        gyroList.clear()
        elapsedTimeList.clear()
        rawTimeList.clear()

    }

    private fun changeDataText(text: String, colour: String) {
        this@DrawingView.rootView.findViewById<TextView>(R.id.dataText).text = text
        this@DrawingView.rootView.findViewById<TextView>(R.id.dataText).setTextColor(Color.parseColor(colour))
    }

    private fun addData() {
        xDrawingData.add(motionTouchEventX)
        yDrawingData.add(motionTouchEventY)
        //this@DrawingView.rootView.findViewById<TextView>(R.id.testView).text = accSensorData.copyOf().contentToString()
        accList.add(accSensorData.copyOf())
        gyroList.add(gyroSensorData.copyOf())

    }

}