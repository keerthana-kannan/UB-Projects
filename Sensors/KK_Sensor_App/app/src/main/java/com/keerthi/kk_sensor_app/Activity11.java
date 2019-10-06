package com.keerthi.kk_sensor_app;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

public class Activity11 extends AppCompatActivity implements SensorEventListener {
    private TextView steps;
    private Sensor myStep;
    private SensorManager sensorM;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_11);

        steps = (TextView)findViewById(R.id.steps);
        sensorM = (SensorManager) getSystemService(SENSOR_SERVICE);

        myStep = sensorM.getDefaultSensor(Sensor.TYPE_STEP_COUNTER);
        if(myStep != null){

            sensorM.registerListener(this, myStep, SensorManager.SENSOR_DELAY_UI);
        }
        else{
            steps.setText("Step Counter Not Supported");
        }
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {

        Sensor sensor = sensorEvent.sensor;
        if(sensor.getType() == Sensor.TYPE_STEP_COUNTER){

            steps.setText("Steps: " + String.valueOf(sensorEvent.values[0]));

        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
}
