package com.robotomy.temi

import android.os.Bundle
import android.util.Log
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import android.widget.ProgressBar
import android.view.View
import java.net.URL

class MainActivity : AppCompatActivity() {

    private val TEMI_URL_BASE = "http://10.0.2.2:8080"  // Temi 서버 IP와 포트로 변경 예정

    private lateinit var progressBar: ProgressBar //progressbar 추가

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val callButton = findViewById<Button>(R.id.button_call_temi)
        val moveButton = findViewById<Button>(R.id.button_move_temi)
        progressBar = findViewById(R.id.progress_bar)

        callButton.setOnClickListener {
            Log.d("Temi", "버튼 클릭됨: /speak")
            sendHttpRequest("/speak")
        }

        moveButton.setOnClickListener {
            Log.d("Temi", "버튼 클릭됨: /move")
            sendHttpRequest("/move")
        }
    }

    private fun sendHttpRequest(endpoint: String) {
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
            }finally {
                runOnUiThread {
                    progressBar.visibility = View.GONE  // 요청 끝나면 로딩 숨김
                }
            }
        }.start()
    }
}
