package com.robotomy.temi;

import com.journeyapps.barcodescanner.CaptureActivity;
import android.content.pm.ActivityInfo;
import android.os.Bundle;

public class MyCaptureActivity extends CaptureActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
    }
}
