package com.keerthi.kk_sensor_app;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.widget.TextView;

public class Activity2 extends AppCompatActivity implements SensorEventListener{

    private TextView xacc,yacc,zacc,gravity;
    private Sensor myAcc;
    private SensorManager sensorM;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_2);

        xacc = (TextView)findViewById(R.id.xacc);
        yacc = (TextView)findViewById(R.id.yacc);
        zacc = (TextView)findViewById(R.id.zacc);
        gravity = (TextView)findViewById(R.id.gravity);

        sensorM = (SensorManager) getSystemService(SENSOR_SERVICE); //create sensor manager

        myAcc = sensorM.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        if(myAcc != null){

            sensorM.registerListener(this, myAcc,SensorManager.SENSOR_DELAY_NORMAL);
        }
        else{
            xacc.setText("Accelerometer Not Supported");
        }
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {

        Sensor sensor = sensorEvent.sensor;
        if (sensor.getType() == Sensor.TYPE_ACCELEROMETER) {

            String s1 = String.format("%.2f", sensorEvent.values[0]);
            String s2 = String.format("%.2f", sensorEvent.values[1]);
            String s3 = String.format("%.2f", sensorEvent.values[2]);
            xacc.setText("X value \n " + s1);
            yacc.setText("Y value \n " + s2);
            zacc.setText("Z value \n" + s3);
            double gravityvalue = Math.sqrt(sensorEvent.values[0] * sensorEvent.values[0] + sensorEvent.values[1] * sensorEvent.values[1] + sensorEvent.values[2] * sensorEvent.values[2] );
            String s4 = String.format("%.2f", gravityvalue);
            gravity.setText("Gravity \n " + s4);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
}
