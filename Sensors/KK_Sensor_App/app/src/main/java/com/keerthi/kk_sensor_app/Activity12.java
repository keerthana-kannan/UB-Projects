package com.keerthi.kk_sensor_app;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

public class Activity12 extends AppCompatActivity implements SensorEventListener {
    private TextView orient;
    private Sensor myorient;
    private SensorManager sensorM;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_12);

        orient = (TextView)findViewById(R.id.orient);

        sensorM = (SensorManager) getSystemService(SENSOR_SERVICE);
        myorient = sensorM.getDefaultSensor(Sensor.TYPE_ORIENTATION);
        if(myorient != null){

            sensorM.registerListener(this, myorient, SensorManager.SENSOR_DELAY_NORMAL);
        }
        else{
            orient.setText("Orientation Sensor Not Supported");
        }
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        Sensor sensor = sensorEvent.sensor;
        if(sensor.getType() == Sensor.TYPE_ORIENTATION)
        {

            if(sensorEvent.values[0]>=348.75 && sensorEvent.values[0]<11.25)
            {
                orient.setText("Direction is North");
            }
            else
            if(sensorEvent.values[0]>=11.25 && sensorEvent.values[0]<33.75)
            {
                orient.setText("Direction is North-NorthEast");
            }
            else
            if(sensorEvent.values[0]>=33.75 && sensorEvent.values[0]<56.25)
            {
                orient.setText("Direction is NorthEast");
            }
            else
            if(sensorEvent.values[0]>=56.25 && sensorEvent.values[0]<78.75)
            {
                orient.setText("Direction is East-NorthEast");
            }
            else
            if(sensorEvent.values[0]>=78.75 && sensorEvent.values[0]<101.25)
            {
                orient.setText("Direction is East");
            }
            else
            if(sensorEvent.values[0]>=101.25 && sensorEvent.values[0]<123.75)
            {
                orient.setText("Direction is East-SouthEast");
            }
            else if(sensorEvent.values[0]>=123.75 && sensorEvent.values[0]<146.25)
            {
                orient.setText("Direction is SouthEast");
            }
            else if(sensorEvent.values[0]>=146.25 && sensorEvent.values[0]<168.75)
            {
                orient.setText("Direction is South-SouthEast");
            }
            else if(sensorEvent.values[0]>=168.75 && sensorEvent.values[0]<191.25)
            {
                orient.setText("Direction is South");
            }
            else if(sensorEvent.values[0]>=191.25 && sensorEvent.values[0]<213.75)
            {
                orient.setText("Direction is South-SouthWest");
            }
            else if(sensorEvent.values[0]>=213.75 && sensorEvent.values[0]<236.25)
            {
                orient.setText("Direction is SouthWest");
            }
            else if(sensorEvent.values[0]>=236.25 && sensorEvent.values[0]<258.75)
            {
                orient.setText("Direction is West-SouthWest");
            }
            else if(sensorEvent.values[0]>=258.75 && sensorEvent.values[0]<281.25)
            {
                orient.setText("Direction is West");
            }
            else if(sensorEvent.values[0]>=281.25 && sensorEvent.values[0]<303.75)
            {
                orient.setText("Direction is West-NorthWest");
            }
            else if(sensorEvent.values[0]>=303.75 && sensorEvent.values[0]<326.25)
            {
                orient.setText("Direction is NorthWest");
            }
            else if(sensorEvent.values[0]>=326.25 && sensorEvent.values[0]<348.75)
            {
                orient.setText("Direction is North-NorthWest");
            }


        }

    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
}
