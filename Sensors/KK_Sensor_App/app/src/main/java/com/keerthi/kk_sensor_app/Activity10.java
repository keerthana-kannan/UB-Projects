package com.keerthi.kk_sensor_app;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

public class Activity10 extends AppCompatActivity implements SensorEventListener {

    private TextView xlinear, ylinear, zlinear, avg;
    private Sensor myLinear;
    private SensorManager sensorM;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_10);

        xlinear = (TextView)findViewById(R.id.xlinear);
        ylinear = (TextView)findViewById(R.id.ylinear);
        zlinear = (TextView)findViewById(R.id.zlinear);
        avg=  (TextView)findViewById(R.id.avg);

        sensorM = (SensorManager) getSystemService(SENSOR_SERVICE);

        myLinear = sensorM.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        if(myLinear != null){

            sensorM.registerListener(this, myLinear, SensorManager.SENSOR_DELAY_NORMAL);
        }
        else{
            xlinear.setText("Linear Accelerometer Not Supported");
        }

    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {

        Sensor sensor = sensorEvent.sensor;
        if(sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION){

            String c1 = String.format("%.2f", sensorEvent.values[0]);
            String c2 = String.format("%.2f", sensorEvent.values[1]);
            String c3 = String.format("%.2f", sensorEvent.values[2]);
            xlinear.setText("X value \n " + c1);
            ylinear.setText("Y value \n " + c2);
            zlinear.setText("Z value \n" + c3);
            double avgvalue = Math.sqrt(sensorEvent.values[0] * sensorEvent.values[0] + sensorEvent.values[1] * sensorEvent.values[1] + sensorEvent.values[2] * sensorEvent.values[2] );
            String c4 = String.format("%.2f", avgvalue);
            avg.setText("Average \n" + c4);

        }

    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
}
