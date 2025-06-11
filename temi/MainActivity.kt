package com.robotomy.temi

import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.journeyapps.barcodescanner.ScanContract
import com.journeyapps.barcodescanner.ScanOptions
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL

class MainActivity : AppCompatActivity() {

    private val TEMI_URL_BASE = "http://10.0.2.2:8080"
    private lateinit var progressBar: ProgressBar
    private lateinit var qrResultText: TextView

    // QR 스캔 기능 추가
    private val qrScanLauncher = registerForActivityResult(ScanContract()) { result ->
        if (result.contents != null) {
            val command = result.contents.trim()
            Log.d("Temi", "QR 인식됨: $command")

            runOnUiThread {
                qrResultText.text = "QR 인식됨: $command"
            }

            if (command == "speak") {
                sendHttpRequest("/speak")
            } else {
                Log.d("Temi", "지원되지 않는 QR 명령어: $command")
            }
        } else {
            Log.d("Temi", "QR 스캔 취소됨")
            runOnUiThread {
                qrResultText.text = "QR 스캔이 취소되었습니다"
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val callButton = findViewById<Button>(R.id.button_call_temi)
        val moveButton = findViewById<Button>(R.id.button_move_temi)
        val qrButton = findViewById<Button>(R.id.button_qr_scan)
        progressBar = findViewById(R.id.progress_bar)
        qrResultText = findViewById(R.id.text_qr_result)

        callButton.setOnClickListener {
            sendHttpRequest("/speak")
        }

        moveButton.setOnClickListener {
            sendHttpRequest("/move")
        }

        qrButton.setOnClickListener {
            val options = ScanOptions()
            options.setPrompt("QR 코드를 스캔하세요")
            options.setBeepEnabled(true)
            options.setBarcodeImageEnabled(true)

            // 세로 고정 및 커스텀 액티비티 지정
            options.setOrientationLocked(true)
            options.setCaptureActivity(MyCaptureActivity::class.java)

            qrScanLauncher.launch(options)
        }
    }

    private fun sendHttpRequest(endpoint: String) {
        runOnUiThread {
            progressBar.visibility = View.VISIBLE
        }

        Thread {
            try {
                val url = URL(TEMI_URL_BASE + endpoint)
                val conn = url.openConnection() as HttpURLConnection
                conn.requestMethod = "GET"
                conn.connectTimeout = 5000
                conn.readTimeout = 5000

                val code = conn.responseCode
                val reader = BufferedReader(InputStreamReader(conn.inputStream))
                val response = reader.readText()

                Log.d("TemiUserApp", "응답 코드: $code, 응답: $response")

                reader.close()
                conn.disconnect()
            } catch (e: Exception) {
                Log.e("TemiUserApp", "에러 발생: ${e.message}")
            } finally {
                runOnUiThread {
                    progressBar.visibility = View.GONE
                }
            }
        }.start()
    }
}
