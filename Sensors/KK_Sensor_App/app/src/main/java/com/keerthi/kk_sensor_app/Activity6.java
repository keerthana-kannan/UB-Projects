package com.keerthi.kk_sensor_app;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

public class Activity6 extends AppCompatActivity implements SensorEventListener {

    private TextView Light;
    private Sensor myLight;
    private SensorManager sensorM;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_6);
        Light = (TextView)findViewById(R.id.Light);

        sensorM = (SensorManager) getSystemService(SENSOR_SERVICE);
        myLight = sensorM.getDefaultSensor(Sensor.TYPE_LIGHT);
        if(myLight != null){

            sensorM.registerListener(this, myLight, SensorManager.SENSOR_DELAY_NORMAL);
        }
        else{
            Light.setText("Light Sensor Not Supported");
        }
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {

        Sensor sensor = sensorEvent.sensor;

        if(sensor.getType() == Sensor.TYPE_LIGHT){

            Light.setText("Light value: " + sensorEvent.values[0]);

        }

    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
}
