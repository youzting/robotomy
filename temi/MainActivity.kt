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
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.PrintWriter
import java.net.Socket

class MainActivity : AppCompatActivity() {

    private val SERVER_IP = "192.168.0.173"
    private val SERVER_PORT = 12345

    private var socket: Socket? = null
    private var out: PrintWriter? = null
    private var input: BufferedReader? = null

    private lateinit var progressBar: ProgressBar
    private lateinit var qrResultText: TextView

    // QR 코드 스캔 런처
    private val qrScanLauncher = registerForActivityResult(ScanContract()) { result ->
        if (result.contents != null) {
            val command = result.contents.trim()
            Log.d("Temi", "QR 인식됨: $command")

            runOnUiThread {
                qrResultText.text = "QR 인식됨: $command"
            }

            when (command) {
                "speak" -> {
                    Log.d("Temi", "QR 명령: speak → sendTcpMessage 호출됨")
                    sendTcpMessage("speak")
                }
                "move" -> {
                    Log.d("Temi", "QR 명령: move → sendTcpMessage 호출됨")
                    sendTcpMessage("move")
                }
                else -> {
                    Log.d("Temi", "지원되지 않는 QR 명령어: $command")
                    runOnUiThread {
                        qrResultText.text = "지원되지 않는 명령어: $command"
                    }
                }
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
            Log.d("Temi", "버튼 클릭: speak → sendTcpMessage 호출됨")
            sendTcpMessage("speak")
        }

        moveButton.setOnClickListener {
            Log.d("Temi", "버튼 클릭: move → sendTcpMessage 호출됨")
            sendTcpMessage("move")
        }

        qrButton.setOnClickListener {
            val options = ScanOptions()
            options.setPrompt("QR 코드를 스캔하세요")
            options.setBeepEnabled(true)
            options.setBarcodeImageEnabled(true)
            options.setOrientationLocked(true)
            options.setCaptureActivity(MyCaptureActivity::class.java)

            Log.d("Temi", "QR 스캔 시작")
            qrScanLauncher.launch(options)
        }

        // 서버 소켓 연결 시도
        connectToServer()
    }

    private fun connectToServer() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                Log.d("TemiUserApp", "서버에 연결 시도: $SERVER_IP:$SERVER_PORT")
                socket = Socket(SERVER_IP, SERVER_PORT)
                out = PrintWriter(socket!!.getOutputStream(), true)
                input = BufferedReader(InputStreamReader(socket!!.getInputStream()))

                runOnUiThread {
                    qrResultText.text = "서버 연결 성공"
                    Log.d("TemiUserApp", "서버 연결 성공")
                }
            } catch (e: Exception) {
                runOnUiThread {
                    qrResultText.text = "서버 연결 실패: ${e.message}"
                    Log.e("TemiUserApp", "서버 연결 실패", e)
                }
            }
        }
    }

    private fun sendTcpMessage(message: String) {
        Log.d("TemiUserApp", "sendTcpMessage 진입: $message")

        CoroutineScope(Dispatchers.IO).launch {
            try {
                runOnUiThread { progressBar.visibility = View.VISIBLE }

                if (socket == null || socket!!.isClosed) {
                    runOnUiThread {
                        qrResultText.text = "서버와 연결이 끊어졌습니다. 다시 연결합니다..."
                        Log.d("TemiUserApp", "서버 연결 재시도")
                    }
                    connectToServer()
                    delay(1000) // 연결이 될 때까지 잠시 대기 (실사용 시 더 정교하게 처리)
                }

                out?.println(message)
                out?.flush()

                val response = input?.readLine() ?: "응답 없음"
                Log.d("TemiUserApp", "서버 응답: $response")

                runOnUiThread {
                    qrResultText.text = "서버 응답: $response"
                }

            } catch (e: Exception) {
                Log.e("TemiUserApp", "TCP 통신 에러: ${e.message}", e)
                runOnUiThread {
                    qrResultText.text = "통신 에러: ${e.message}"
                }
            } finally {
                runOnUiThread { progressBar.visibility = View.GONE }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            socket?.close()
            Log.d("TemiUserApp", "소켓 종료 완료")
        } catch (e: Exception) {
            Log.e("TemiUserApp", "소켓 종료 중 에러", e)
        }
    }
}
