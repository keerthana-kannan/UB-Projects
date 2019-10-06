package com.keerthi.kk_sensor_app;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

public class Activity7 extends AppCompatActivity implements SensorEventListener {

    private TextView pressure;
    private Sensor myPressure;
    private SensorManager sensorM;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_7);
        pressure = (TextView)findViewById(R.id.pressure);
        sensorM = (SensorManager) getSystemService(SENSOR_SERVICE);

        myPressure = sensorM.getDefaultSensor(Sensor.TYPE_PRESSURE);
        if(myPressure != null){

            sensorM.registerListener(this, myPressure, SensorManager.SENSOR_DELAY_NORMAL);
        }
        else{
            pressure.setText("Pressure Sensor Not Supported");
        }
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {

        Sensor sensor = sensorEvent.sensor;
        if(sensor.getType() == Sensor.TYPE_PRESSURE){

            pressure.setText("Pressure value: " + sensorEvent.values[0]);

        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
}
