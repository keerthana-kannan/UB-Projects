package com.keerthi.kk_sensor_app;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

public class Activity4 extends AppCompatActivity implements SensorEventListener {

    private TextView xMagno,yMagno,zMagno,average;
    private Sensor myMagno;
    private SensorManager sensorM;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_4);

        xMagno = (TextView)findViewById(R.id.xMagno);
        yMagno = (TextView)findViewById(R.id.yMagno);
        zMagno = (TextView)findViewById(R.id.zMagno);
        average = (TextView) findViewById(R.id.average);

        sensorM = (SensorManager) getSystemService(SENSOR_SERVICE); //create sensor manager

        myMagno = sensorM.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        if(myMagno != null){

            sensorM.registerListener(this, myMagno, SensorManager.SENSOR_DELAY_NORMAL);
        }
        else{
            xMagno.setText("Magnetometer Not Supported");
        }
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        Sensor sensor = sensorEvent.sensor;
        if(sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD){

            String a1 = String.format("%.2f", sensorEvent.values[0]);
            String a2 = String.format("%.2f", sensorEvent.values[1]);
            String a3 = String.format("%.2f", sensorEvent.values[2]);
            xMagno.setText("X value \n " + a1);
            yMagno.setText("Y value \n " + a2);
            zMagno.setText("Z value \n" + a3);
            double average1 = Math.sqrt(sensorEvent.values[0] * sensorEvent.values[0] + sensorEvent.values[1] * sensorEvent.values[1] + sensorEvent.values[2] * sensorEvent.values[2] );
            String a4 = String.format("%.2f", average1);
            average.setText("Average \n " + a4);

        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
}
