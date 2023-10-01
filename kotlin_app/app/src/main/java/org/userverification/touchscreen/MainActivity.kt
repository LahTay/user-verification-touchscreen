package org.userverification.touchscreen

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.view.WindowManager
import android.widget.TextView
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.updateLayoutParams

//TODO: Are we using correct sensor data?
//TODO: Sensor in runnable 60 hz?
//TODO: Graphic makeover?

class MainActivity : AppCompatActivity(), SensorEventListener {

    private lateinit var permissionLauncher: ActivityResultLauncher<Array<String>>
    private var isReadPermissionGranted = false
    private var isWritePermissionGranted = false

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    //private var accData: FloatArray? = null
    //private var gyroData: FloatArray? = null
    private lateinit var drawingView: DrawingView

    @SuppressLint("SourceLockedOrientationActivity")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) // keeps our screen on while using the app
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT

        val canvasSize = getCanvasSize()

        drawingView = findViewById<DrawingView>(R.id.drawing_view)


        drawingView.updateLayoutParams {
            height = canvasSize
            width= canvasSize
        }

        drawingView.thisViewSize = canvasSize

        // get permissionLauncher to register our required permissions for saving data to our phone storage
        permissionLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()){ permissions ->
            isReadPermissionGranted = permissions[Manifest.permission.READ_EXTERNAL_STORAGE] ?: isReadPermissionGranted
            isWritePermissionGranted = permissions[Manifest.permission.WRITE_EXTERNAL_STORAGE] ?: isWritePermissionGranted
        }

        // run function which updates our permission status
        requestPermission()

        // get an instance of the SensorManager to get our sensors
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager

        // check if the device has an accelerometer and gyroscope
        if (sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) != null && sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE) != null) {
            // get our sensors
            accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
            gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

            drawingView.accProperties.add(SensorProperties(accelerometer!!.maximumRange, accelerometer!!.resolution, accelerometer!!.minDelay, accelerometer!!.maxDelay))
            drawingView.gyroProperties.add(SensorProperties(gyroscope!!.maximumRange, gyroscope!!.resolution, gyroscope!!.minDelay, gyroscope!!.maxDelay))

        } else {
            // missing sensors - we could also close the app here
            this@MainActivity.findViewById<TextView>(R.id.dataText).text = getString(R.string.missing_sensors)
        }

    }

    override fun onResume() {
        super.onResume()
        // register the sensor listeners on resume
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_FASTEST)
        sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_FASTEST)
    }

    override fun onPause() {
        super.onPause()
        // unregister the listener of our sensors
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        // get values from our sensors and display them - we get our data continuously as it changes
        if (event.sensor == accelerometer) {
            //accData = event.values
            drawingView.accSensorData = event.values
        } else if (event.sensor == gyroscope) {
            //gyroData = event.values
            drawingView.gyroSensorData = event.values
        }
        //this@MainActivity.findViewById<TextView>(R.id.testView).text = "Accelerometer: " + accData.contentToString() +"\nGyroscope: " + gyroData.contentToString()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // this function is here, because it will throw error otherwise
    }

    private fun requestPermission(){
        // simple function that checks for permissions and updates them as needed
        isReadPermissionGranted = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.READ_EXTERNAL_STORAGE
        ) == PackageManager.PERMISSION_GRANTED

        isWritePermissionGranted = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        ) == PackageManager.PERMISSION_GRANTED

        val permissionRequest : MutableList<String> = ArrayList()

        if(!isReadPermissionGranted){
            permissionRequest.add(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
        if(!isWritePermissionGranted){
            permissionRequest.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }

        if(permissionRequest.isNotEmpty()){
            permissionLauncher.launch(permissionRequest.toTypedArray())
        }
    }

    @SuppressLint("DiscouragedApi")
    private fun getSystemDimension(name: String): Int {
        val resourceId = resources.getIdentifier(name, "dimen", "android")
        if (resourceId > 0) {
            return resources.getDimensionPixelSize(resourceId)
        }
        return 0
    }

    private fun getCanvasSize(): Int {
        val displayMetrics = resources.displayMetrics
        val widthPixels = displayMetrics.widthPixels
        val heightPixels = displayMetrics.heightPixels
        val navigationBarHeight = getSystemDimension("navigation_bar_height")
        val statusBarHeight = getSystemDimension("status_bar_height")

        var size = (widthPixels*9)/10
        val trueHeight = heightPixels-(navigationBarHeight+statusBarHeight)

        if ((trueHeight/2)<size) {
            size = (trueHeight*9)/20
        }
        return size
    }

}