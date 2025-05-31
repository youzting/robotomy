package com.robotomy.temi

import android.os.Bundle
import android.util.Log
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import java.io.*
import java.net.Socket

class MainActivity : AppCompatActivity() {

    private val TEMI_IP = "10.0.2.2"//안드로이드 스튜디오 에뮬레이터 ip주소
    private val TEMI_PORT = 12345//임의 테스트 포트

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val callButton = findViewById<Button>(R.id.button_call_temi)
        Log.d("TemiUserApp", "callButton is null? ${callButton == null}")

        callButton.setOnClickListener {
            Log.d("TemiUserApp", "버튼 클릭됨")
            sendCallTemi()
        }
    }

    private fun sendCallTemi() {
        Log.d("TemiUserApp", "sendCallTemi 시작")
        Thread {
            try {
                val inet4Address = java.net.Inet4Address.getByName(TEMI_IP) as java.net.Inet4Address
                val socket = java.net.Socket()
                socket.connect(java.net.InetSocketAddress(inet4Address, TEMI_PORT), 5000) // 5초 타임아웃

                val writer = PrintWriter(socket.getOutputStream(), true)
                val reader = BufferedReader(InputStreamReader(socket.getInputStream()))

                writer.println("CALL_TEMI")
                val response = reader.readLine()
                Log.d("TemiUserApp", "응답 받음: $response")

                socket.close()
                Log.d("TemiUserApp", "소켓 닫음")
            } catch (e: Exception) {
                Log.e("TemiUserApp", "에러: ${e.message}")
            }
        }.start()
    }

}
