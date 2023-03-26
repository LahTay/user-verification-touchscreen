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


//TODO: Drawing
//TODO: Saving sensor data while drawing

class MainActivity : AppCompatActivity(), SensorEventListener {

    private lateinit var permissionLauncher: ActivityResultLauncher<Array<String>>
    private var isReadPermissionGranted = false
    private var isWritePermissionGranted = false

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    private var accData: FloatArray? = null
    private var gyroData: FloatArray? = null

    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) // keeps our screen on while using the app
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT

        val canvasSize = getCanvasSize()

        val drawingView = findViewById<DrawingView>(R.id.drawing_view)

        drawingView.updateLayoutParams {
            height = canvasSize
            width= canvasSize
        }

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
        } else {
            // missing sensors - we could also close the app here
            this@MainActivity.findViewById<TextView>(R.id.dataText).text = "Missing sensors!"
        }

    }

    override fun onResume() {
        super.onResume()
        // register the sensor listeners on resume
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL)
        sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_NORMAL)
    }

    override fun onPause() {
        super.onPause()
        // unregister the listener of our sensors
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        // get values from our sensors and display them - we get our data continuously as it changes
        if (event.sensor == accelerometer) {
            accData = event.values
        } else if (event.sensor == gyroscope) {
            gyroData = event.values
        }

        this@MainActivity.findViewById<TextView>(R.id.dataText).text = "Accelerometer: " + accData.contentToString() + "\nGyroscope: " + gyroData.contentToString()
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

    private fun getNavigationBarHeight(): Int {
        val resourceId = resources.getIdentifier("navigation_bar_height", "dimen", "android")
        if (resourceId > 0) {
            return resources.getDimensionPixelSize(resourceId)
        }
        return 0
    }

    private fun getStatusBarHeight(): Int {
        val resourceId = resources.getIdentifier("status_bar_height", "dimen", "android")
        if (resourceId > 0) {
            return resources.getDimensionPixelSize(resourceId)
        }
        return 0
    }

    private fun getCanvasSize(): Int {
        val displayMetrics = resources.displayMetrics
        val widthPixels = displayMetrics.widthPixels
        val heightPixels = displayMetrics.heightPixels

        var size = (widthPixels*9)/10
        val trueHeight = heightPixels-(getNavigationBarHeight()+getStatusBarHeight())

        if ((trueHeight/2)<size) {
            size = (trueHeight*9)/20
        }
        return size
    }

}