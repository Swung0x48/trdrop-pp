package io.github.swung0x48.trdrop

import android.content.Intent
import android.os.Bundle
import android.text.format.DateUtils
import android.view.View
import android.view.WindowManager
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.progressindicator.LinearProgressIndicator

/**
 * Choose a video, analyze it, open earlier analyses.
 *
 * For tests an analysis can be started with intent extras:
 *   adb shell am start -n io.github.swung0x48.trdrop/.MainActivity --es video <path> --ez autostart true
 * with the optional --ez async / motion_video / arrows / cpu / gpu (analyze on the CPU / GPU) and --es args "<more arguments for trdrop-pp>".
 */
class MainActivity : AppCompatActivity(), AnalysisRunner.Listener {
    private var source: String? = null

    private lateinit var videoName: TextView
    private lateinit var optionAsync: MaterialSwitch
    private lateinit var optionMotionVideo: MaterialSwitch
    private lateinit var optionArrows: MaterialSwitch
    private lateinit var start: MaterialButton
    private lateinit var pick: MaterialButton
    private lateinit var progressGroup: View
    private lateinit var progressTitle: TextView
    private lateinit var progress: LinearProgressIndicator
    private lateinit var progressText: TextView
    private lateinit var message: TextView
    private lateinit var history: LinearLayout
    private lateinit var historyEmpty: View
    private lateinit var processor: com.google.android.material.button.MaterialButtonToggleGroup
    private var extraArguments: List<String> = emptyList()

    private val pickVideo = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) select(uri.toString())
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.root)) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            WindowInsetsCompat.CONSUMED
        }
        videoName = findViewById(R.id.video_name)
        optionAsync = findViewById(R.id.option_async)
        optionMotionVideo = findViewById(R.id.option_motion_video)
        optionArrows = findViewById(R.id.option_arrows)
        start = findViewById(R.id.start)
        processor = findViewById(R.id.processor)
        pick = findViewById(R.id.pick)
        progressGroup = findViewById(R.id.progress_group)
        progressTitle = findViewById(R.id.progress_title)
        progress = findViewById(R.id.progress)
        progressText = findViewById(R.id.progress_text)
        message = findViewById(R.id.message)
        history = findViewById(R.id.history)
        historyEmpty = findViewById(R.id.history_empty)

        pick.setOnClickListener { pickVideo.launch(arrayOf("video/*")) }
        start.setOnClickListener { startAnalysis() }
        findViewById<View>(R.id.cancel).setOnClickListener { AnalysisRunner.cancel() }
        optionMotionVideo.setOnCheckedChangeListener { _, checked -> optionArrows.isEnabled = checked }

        savedInstanceState?.getString("source")?.let(::select)
        if (savedInstanceState == null) handleIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: Intent) {
        // a video opened with the app (or, for tests, given as an extra)
        val video = intent.getStringExtra("video") ?: intent.data?.toString() ?: return
        select(video)
        optionAsync.isChecked = intent.getBooleanExtra("async", false)
        optionMotionVideo.isChecked = intent.getBooleanExtra("motion_video", true)
        optionArrows.isChecked = intent.getBooleanExtra("arrows", false)
        processor.check(
            when {
                intent.getBooleanExtra("cpu", false) -> R.id.processor_cpu
                intent.getBooleanExtra("gpu", false) -> R.id.processor_gpu
                else -> R.id.processor_auto
            }
        )
        extraArguments = intent.getStringExtra("args")?.split(' ')?.filter { it.isNotEmpty() } ?: emptyList()
        if (intent.getBooleanExtra("autostart", false)) startAnalysis()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putString("source", source)
    }

    override fun onResume() {
        super.onResume()
        AnalysisRunner.listener = this
        updateRunning()
        if (AnalysisRunner.running) onProgress(AnalysisRunner.lastFrames, AnalysisRunner.lastProgress)
        showHistory()
    }

    override fun onPause() {
        super.onPause()
        if (AnalysisRunner.listener === this) AnalysisRunner.listener = null
    }

    private fun select(value: String) {
        source = value
        videoName.text = AnalysisRunner.displayName(this, value)
        updateRunning()
    }

    private fun startAnalysis() {
        val value = source ?: return
        message.visibility = View.GONE
        val request = AnalysisRequest(
            source = value,
            asyncCapture = optionAsync.isChecked,
            motionVideo = optionMotionVideo.isChecked,
            arrowScale = if (optionArrows.isChecked) 8.0 else 1.0,
            decode = when (processor.checkedButtonId) {
                R.id.processor_gpu -> "vulkan"
                R.id.processor_cpu -> "ffmpeg"
                else -> "auto"
            },
            extraArguments = extraArguments,
        )
        if (AnalysisRunner.start(this, request)) updateRunning()
    }

    private fun updateRunning() {
        val running = AnalysisRunner.running
        start.isEnabled = !running && source != null
        pick.isEnabled = !running
        progressGroup.visibility = if (running) View.VISIBLE else View.GONE
        if (running) {
            progressTitle.text = getString(R.string.analyzing, AnalysisRunner.runningName)
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        } else {
            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }

    override fun onProgress(frames: Long, progress: Double) {
        this.progress.isIndeterminate = progress < 0
        if (progress >= 0) {
            this.progress.setProgressCompat((progress * 1000).toInt(), true)
            progressText.text = getString(R.string.progress_frames, frames, progress * 100.0)
        } else {
            progressText.text = getString(R.string.progress_frames_unknown, frames)
        }
    }

    override fun onFinished(analysis: Analysis, exitCode: Int) {
        updateRunning()
        showHistory()
        if (exitCode == 0) {
            startActivity(Intent(this, ResultActivity::class.java).putExtra("id", analysis.id))
        } else {
            val error = Summary(analysis.logLines()).error
            message.text = if (error == "cancelled") getString(R.string.cancelled) else getString(R.string.failed, error)
            message.visibility = View.VISIBLE
            analysis.directory.deleteRecursively()
            showHistory()
        }
    }

    private fun showHistory() {
        history.removeAllViews()
        val analyses = Analysis.list(this).filter { it.succeeded }
        historyEmpty.visibility = if (analyses.isEmpty()) View.VISIBLE else View.GONE
        val density = resources.displayMetrics.density
        for (analysis in analyses) {
            val summary = Summary(analysis.logLines())
            val card = MaterialCardView(this, null, com.google.android.material.R.attr.materialCardViewFilledStyle).apply {
                setCardBackgroundColor(getColor(R.color.chart_surface))
                radius = 16 * density
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                    topMargin = (8 * density).toInt()
                }
            }
            val column = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                val padding = (16 * density).toInt()
                setPadding(padding, (12 * density).toInt(), padding, (12 * density).toInt())
            }
            column.addView(TextView(this).apply {
                text = analysis.name
                setTextColor(getColor(R.color.chart_text_primary))
                textSize = 15f
                isSingleLine = true
                ellipsize = android.text.TextUtils.TruncateAt.MIDDLE
            })
            val numbers = ArrayList<String>()
            // with lost pictures counted in, the framerate is the one of the motion line
            val framerate = if ((summary.lostPictures ?: 0.0) > 0.0) summary.motionFps else summary.fpsAverage
            framerate?.let { numbers += getString(R.string.value_fps, it) }
            summary.perceivedFps?.let { numbers += getString(R.string.stat_perceived) + " " + getString(R.string.value_fps, it) }
            numbers += DateUtils.getRelativeTimeSpanString(analysis.startedAt, System.currentTimeMillis(), DateUtils.MINUTE_IN_MILLIS).toString()
            column.addView(TextView(this).apply {
                text = numbers.joinToString(" · ")
                setTextColor(getColor(R.color.chart_text_secondary))
                textSize = 13f
            })
            card.addView(column)
            card.setOnClickListener { startActivity(Intent(this, ResultActivity::class.java).putExtra("id", analysis.id)) }
            card.setOnLongClickListener {
                AlertDialog.Builder(this)
                    .setMessage(R.string.delete_question)
                    .setPositiveButton(R.string.delete) { _, _ ->
                        analysis.directory.deleteRecursively()
                        showHistory()
                    }
                    .setNegativeButton(android.R.string.cancel, null)
                    .show()
                true
            }
            history.addView(card)
        }
    }
}
