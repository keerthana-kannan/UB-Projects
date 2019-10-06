package com.keerthi.kk_sensor_app;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.Button;
import android.view.View;
import android.content.Intent;

public class MainActivity extends AppCompatActivity implements View.OnClickListener  {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button button1 = findViewById(R.id.button1);
        Button button2 = findViewById(R.id.button2);
        Button button3 = findViewById(R.id.button3);
        Button button4 = findViewById(R.id.button4);
        Button button5 = findViewById(R.id.button5);
        Button button6 = findViewById(R.id.button6);
        Button button7 = findViewById(R.id.button7);
        Button button8 = findViewById(R.id.button8);
        Button button9 = findViewById(R.id.button9);
        Button button10 = findViewById(R.id.button10);
        Button button11 = findViewById(R.id.button11);
        Button button12 = findViewById(R.id.button12);


        button1.setOnClickListener(this);
        button2.setOnClickListener(this);
        button3.setOnClickListener(this);
        button4.setOnClickListener(this);
        button5.setOnClickListener(this);
        button6.setOnClickListener(this);
        button7.setOnClickListener(this);
        button8.setOnClickListener(this);
        button9.setOnClickListener(this);
        button10.setOnClickListener(this);
        button11.setOnClickListener(this);
        button12.setOnClickListener(this);
    }

    @Override
    public void onClick(View view) {

        switch(view.getId()) {
            case R.id.button1:
                openActivity2();
                break;
            case R.id.button2:
                openActivity3();
                break;
            case R.id.button3:
                openActivity4();
                break;
            case R.id.button4:
                openActivity5();
                break;
            case R.id.button5:
                openActivity6();
                break;
            case R.id.button6:
                openActivity7();
                break;
            case R.id.button7:
                openActivity8();
                break;
            case R.id.button8:
                openActivity9();
                break;
            case R.id.button9:
                openActivity10();
                break;
            case R.id.button10:
                openActivity11();
                break;
            case R.id.button11:
                openActivity12();
                break;
            case R.id.button12:
                openActivity13();
                break;
        }

    }
    public void openActivity2() {

        Intent intent = new Intent(this, Activity2.class);
        startActivity(intent);
    }
    public void openActivity3() {

        Intent intent = new Intent(this, Activity3.class);
        startActivity(intent);
    }
    public void openActivity4() {

        Intent intent = new Intent(this, Activity4.class);
        startActivity(intent);
    }
    public void openActivity5() {

        Intent intent = new Intent(this, Activity5.class);
        startActivity(intent);
    }
    public void openActivity6() {

        Intent intent = new Intent(this, Activity6.class);
        startActivity(intent);
    }
    public void openActivity7() {

        Intent intent = new Intent(this, Activity7.class);
        startActivity(intent);
    }
    public void openActivity8() {

        Intent intent = new Intent(this, Activity8.class);
        startActivity(intent);
    }
    public void openActivity9() {

        Intent intent = new Intent(this, Activity9.class);
        startActivity(intent);
    }
    public void openActivity10() {

        Intent intent = new Intent(this, Activity10.class);
        startActivity(intent);
    }
    public void openActivity11() {

        Intent intent = new Intent(this, Activity11.class);
        startActivity(intent);
    }
    public void openActivity12() {

        Intent intent = new Intent(this, Activity12.class);
        startActivity(intent);
    }
    public void openActivity13() {

        Intent intent = new Intent(this, Activity13.class);
        startActivity(intent);
    }
}
