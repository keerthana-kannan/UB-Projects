package com.keerthi.kk_sensor_app;

import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

public class Activity3 extends AppCompatActivity implements SensorEventListener {

    private TextView proxi;
    private Sensor myProxi;
    private SensorManager sensorM;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_3);

        proxi = (TextView)findViewById(R.id.proxi);

        sensorM = (SensorManager) getSystemService(SENSOR_SERVICE); //create sensor manager

        myProxi = sensorM.getDefaultSensor(Sensor.TYPE_PROXIMITY);
        if(myProxi != null){

            sensorM.registerListener(this, myProxi, SensorManager.SENSOR_DELAY_NORMAL);
        }
        else{
            proxi.setText("Proximity Sensor Not Supported");
        }
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        Sensor sensor = sensorEvent.sensor;
        if (sensor.getType() == Sensor.TYPE_PROXIMITY) {
            if (sensorEvent.values[0] < myProxi.getMaximumRange()) {

                proxi.setText("Too close to the sensor");
                getWindow().getDecorView().setBackgroundColor(Color.DKGRAY);
            } else {
                proxi.setText("Too far from the sensor");
                getWindow().getDecorView().setBackgroundColor(Color.LTGRAY);
            }

        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
}
