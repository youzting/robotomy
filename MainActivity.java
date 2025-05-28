package com.example.temiexample;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import com.robotemi.sdk.Robot;
import com.robotemi.sdk.listeners.OnLocationsUpdatedListener;

import java.util.List;

public class MainActivity extends AppCompatActivity implements OnLocationsUpdatedListener {

    private Robot robot;
    private Button btnSpeak, btnGoTo, btnShowLocations;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        robot = Robot.getInstance();  // Temi SDK 핵심 객체

        btnSpeak = findViewById(R.id.btnSpeak);
        btnGoTo = findViewById(R.id.btnGoTo);
        btnShowLocations = findViewById(R.id.btnShowLocations);

        btnSpeak.setOnClickListener(v -> robot.speak("안녕하세요! 저는 테미입니다."));

        btnGoTo.setOnClickListener(v -> robot.goTo("home base")); // 위치 이름은 Temi에서 등록된 위치여야 함

        btnShowLocations.setOnClickListener(v -> {
            List<String> locations = robot.getLocations();
            Toast.makeText(this, "등록된 위치: " + locations, Toast.LENGTH_LONG).show();
        });
    }

    @Override
    protected void onStart() {
        super.onStart();
        robot.addOnLocationsUpdatedListener(this);
    }

    @Override
    protected void onStop() {
        super.onStop();
        robot.removeOnLocationsUpdatedListener(this);
    }

    @Override
    public void onLocationsUpdated(List<String> locations) {
        // 위치 업데이트 될 때 호출됨
    }
}
