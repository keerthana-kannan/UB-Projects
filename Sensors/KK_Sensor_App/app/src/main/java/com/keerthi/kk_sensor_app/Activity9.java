package com.keerthi.kk_sensor_app;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

public class Activity9 extends AppCompatActivity implements SensorEventListener {

    private TextView Humid;
    private Sensor myHumid;
    private SensorManager sensorM;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_9);
        Humid = (TextView)findViewById(R.id.Humid);

        sensorM = (SensorManager) getSystemService(SENSOR_SERVICE);
        myHumid = sensorM.getDefaultSensor(Sensor.TYPE_RELATIVE_HUMIDITY);
        if(myHumid != null){

            sensorM.registerListener(this, myHumid, SensorManager.SENSOR_DELAY_NORMAL);
        }
        else{
            Humid.setText("Humidity Sensor Not Supported");
        }

    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        Sensor sensor = sensorEvent.sensor;
        if(sensor.getType() == Sensor.TYPE_RELATIVE_HUMIDITY){

            Humid.setText("Relative Humidity: " + sensorEvent.values[0]);

        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
}
