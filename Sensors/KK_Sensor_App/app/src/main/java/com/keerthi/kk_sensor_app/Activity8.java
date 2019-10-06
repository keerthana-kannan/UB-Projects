package com.keerthi.kk_sensor_app;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

public class Activity8 extends AppCompatActivity implements SensorEventListener {

    private TextView temp;
    private Sensor myTemp;
    private SensorManager sensorM;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_8);
        temp = (TextView)findViewById(R.id.temp);
        sensorM = (SensorManager) getSystemService(SENSOR_SERVICE);
        myTemp = sensorM.getDefaultSensor(Sensor.TYPE_AMBIENT_TEMPERATURE);
        if(myTemp != null){

            sensorM.registerListener(this, myTemp, SensorManager.SENSOR_DELAY_NORMAL);
        }
        else{
            temp.setText("Temperature Sensor Not Supported");
        }

    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {

        Sensor sensor = sensorEvent.sensor;

        if(sensor.getType() == Sensor.TYPE_AMBIENT_TEMPERATURE){

            temp.setText("Temperature Value: " + sensorEvent.values[0]);

        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
}
