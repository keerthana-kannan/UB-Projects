package com.keerthi.kk_sensor_app;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

public class Activity5 extends AppCompatActivity implements SensorEventListener {

    private TextView xGyro,yGyro,zGyro,Average;
    private Sensor myGyro;
    private SensorManager sensorM;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_5);

        xGyro = (TextView)findViewById(R.id.xGyro);
        yGyro = (TextView)findViewById(R.id.yGyro);
        zGyro = (TextView)findViewById(R.id.zGyro);
        Average = (TextView)findViewById(R.id.Average);

        sensorM = (SensorManager) getSystemService(SENSOR_SERVICE); //create sensor manager
        myGyro = sensorM.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        if(myGyro != null){

            sensorM.registerListener(this, myGyro, SensorManager.SENSOR_DELAY_NORMAL);
        }
        else{
            xGyro.setText("Gyroscope Not Supported");
        }
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {

        Sensor sensor = sensorEvent.sensor;
        if(sensor.getType() == Sensor.TYPE_GYROSCOPE){


            String b1 = String.format("%.2f", sensorEvent.values[0]);
            String b2 = String.format("%.2f", sensorEvent.values[1]);
            String b3 = String.format("%.2f", sensorEvent.values[2]);
            xGyro.setText("X value \n " + b1);
            yGyro.setText("Y value \n " + b2);
            zGyro.setText("Z value \n" + b3);
            double Averagevalue = Math.sqrt(sensorEvent.values[0] * sensorEvent.values[0] + sensorEvent.values[1] * sensorEvent.values[1] + sensorEvent.values[2] * sensorEvent.values[2] );
            String b4 = String.format("%.2f", Averagevalue);
            Average.setText("Average \n" + b4);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
}
