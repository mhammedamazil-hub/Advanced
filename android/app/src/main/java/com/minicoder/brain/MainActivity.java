package com.minicoder.brain;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.os.Bundle;
import android.util.TypedValue;
import android.view.Gravity;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * MiniCoder Brain — the brain runs ON the phone.
 *
 * The APK carries only the Python/NumPy runtime and this shell. The actual
 * brain (the minicoder/ Python package) is downloaded STRAIGHT FROM GITHUB
 * the first time the user taps "Download brain files". After that everything
 * — chat, memory, rules, skills, sleep cycles — works fully OFFLINE.
 *
 * A "remote server" mode remains as a fallback (brain hosted on a PC),
 * useful when training speed matters.
 */
public class MainActivity extends Activity {

    private static final String PREFS = "brain";
    private static final String KEY_REMOTE = "remote_url";
    private static final String KEY_SRC_READY = "src_ready";
    private static final String DEFAULT_ZIP =
            "https://codeload.github.com/mhammedamazil-hub/Advanced/zip/refs/heads/main";

    private static final int BG = 0xFF0D1117, FG = 0xFFC9D1D9, ACC = 0xFF7C5CFF,
            MUT = 0xFF8B949E, OK = 0xFF3FB950;

    private SharedPreferences prefs;
    private FrameLayout root;
    private WebView web;

    private int dp(int v) {
        return (int) TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, v, getResources().getDisplayMetrics());
    }

    private File srcDir()  { return new File(getFilesDir(), "src"); }
    private File dataDir() { return new File(getFilesDir(), "data"); }

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        root = new FrameLayout(this);
        root.setBackgroundColor(BG);
        setContentView(root);
        if (prefs.getBoolean(KEY_SRC_READY, false)) {
            startLocalBrain();
        } else if (!prefs.getString(KEY_REMOTE, "").isEmpty()) {
            showBrain(prefs.getString(KEY_REMOTE, ""));
            addGear();
        } else {
            showSetup();
        }
    }

    // ---------------------------------------------------------- setup screen
    private void showSetup() {
        LinearLayout col = new LinearLayout(this);
        col.setOrientation(LinearLayout.VERTICAL);
        col.setPadding(dp(28), dp(30), dp(28), dp(20));

        TextView title = new TextView(this);
        title.setText("🧠 MiniCoder Brain");
        title.setTextColor(FG);
        title.setTextSize(26);
        col.addView(title);

        TextView sub = new TextView(this);
        sub.setText("This phone's copy of the brain is downloaded from GitHub "
                + "(the small APK only carries the language runtime). "
                + "After the one-time download, everything works offline.");
        sub.setTextColor(MUT);
        sub.setTextSize(13);
        sub.setPadding(0, dp(8), 0, dp(22));
        col.addView(sub);

        boolean ready = prefs.getBoolean(KEY_SRC_READY, false);
        Button mainBtn = new Button(this);
        mainBtn.setText(ready ? "▶  Start brain" : "⬇  Download brain files from GitHub");
        mainBtn.setTextColor(Color.WHITE);
        mainBtn.setBackgroundColor(ready ? OK : ACC);
        col.addView(mainBtn, matchWrap(dp(18)));

        final TextView progress = new TextView(this);
        progress.setTextColor(MUT);
        progress.setTextSize(12);
        progress.setPadding(0, dp(10), 0, 0);
        col.addView(progress);

        mainBtn.setOnClickListener(v -> {
            if (prefs.getBoolean(KEY_SRC_READY, false)) {
                startLocalBrain();
            } else {
                mainBtn.setEnabled(false);
                downloadBrain(progress);
            }
        });

        TextView or = new TextView(this);
        or.setText("\n— or connect to a brain server (PC / cloud) —");
        or.setTextColor(MUT);
        or.setTextSize(12);
        or.setGravity(Gravity.CENTER);
        col.addView(or);

        final EditText urlIn = new EditText(this);
        urlIn.setHint("http://192.168.1.20:8000");
        urlIn.setHintTextColor(0xFF484F58);
        urlIn.setTextColor(FG);
        urlIn.setSingleLine(true);
        urlIn.setText(prefs.getString(KEY_REMOTE, ""));
        col.addView(urlIn, matchWrap(dp(14)));

        Button remote = new Button(this);
        remote.setText("Connect to server →");
        col.addView(remote, matchWrap(dp(10)));
        remote.setOnClickListener(v -> {
            String url = urlIn.getText().toString().trim();
            if (url.isEmpty()) return;
            if (!url.startsWith("http")) url = "http://" + url;
            prefs.edit().putString(KEY_REMOTE, url).apply();
            root.removeAllViews();
            showBrain(url);
            addGear();
        });

        TextView attr = new TextView(this);
        attr.setText("\nPython runtime © Chaquopy (MIT) · MiniCoder is open source");
        attr.setTextColor(0xFF484F58);
        attr.setTextSize(11);
        col.addView(attr);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(col);
        root.addView(scroll, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT));
    }

    private LinearLayout.LayoutParams matchWrap(int topMarginPx) {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.setMargins(0, topMarginPx, 0, 0);
        return lp;
    }

    // -------------------------------------------------------- github download
    private void downloadBrain(TextView progress) {
        new Thread(() -> {
            try {
                ui(progress, "connecting to github…");
                HttpURLConnection conn = (HttpURLConnection) new URL(DEFAULT_ZIP).openConnection();
                conn.setConnectTimeout(20000);
                conn.setReadTimeout(60000);
                InputStream raw = new BufferedInputStream(conn.getInputStream());

                File out = srcDir();
                deleteRecursive(out);
                out.mkdirs();
                ui(progress, "downloading + extracting…");
                unzipFirstTopLevel(raw, out);
                if (!new File(out, "minicoder/memory/store.py").exists()) {
                    throw new RuntimeException("download did not contain the brain source");
                }
                prefs.edit().putBoolean(KEY_SRC_READY, true).apply();
                runOnUiThread(() -> {
                    root.removeAllViews();
                    startLocalBrain();
                });
            } catch (Exception e) {
                ui(progress, "⚠ download failed: " + e.getMessage()
                        + "\n(check internet, then tap again)");
                runOnUiThread(() -> showSetup());
            }
        }).start();
    }

    /** Extract a github zip (single top-level folder) into dir, stripping it. */
    private void unzipFirstTopLevel(InputStream in, File dir) throws Exception {
        String canon = dir.getCanonicalPath();
        ZipInputStream z = new ZipInputStream(in);
        byte[] buf = new byte[64 * 1024];
        ZipEntry e;
        while ((e = z.getNextEntry()) != null) {
            String name = e.getName();
            int slash = name.indexOf('/');
            if (slash < 0 || slash == name.length() - 1) continue;   // top-level itself
            String rel = name.substring(slash + 1);
            if (rel.isEmpty()) continue;
            File f = new File(dir, rel);
            if (!f.getCanonicalPath().startsWith(canon)) continue;   // zip-slip guard
            if (e.isDirectory()) { f.mkdirs(); continue; }
            f.getParentFile().mkdirs();
            FileOutputStream fo = new FileOutputStream(f);
            int n;
            while ((n = z.read(buf)) > 0) fo.write(buf, 0, n);
            fo.close();
        }
        z.close();
    }

    private void deleteRecursive(File f) {
        if (!f.exists()) return;
        File[] kids = f.listFiles();
        if (kids != null) for (File k : kids) deleteRecursive(k);
        f.delete();
    }

    // --------------------------------------------------------- local brain
    private void startLocalBrain() {
        final TextView status = new TextView(this);
        status.setText("🧠 waking the brain…\n(first wake creates memory & tokenizer — can take a minute on a phone)");
        status.setTextColor(MUT);
        status.setTextSize(14);
        status.setGravity(Gravity.CENTER);
        status.setPadding(dp(20), 0, dp(20), 0);
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.CENTER);
        root.removeAllViews();
        root.addView(status, lp);

        new Thread(() -> {
            try {
                if (!Python.isStarted()) Python.start(new AndroidPlatform(this));
                Python py = Python.getInstance();
                dataDir().mkdirs();
                String msg = py.getModule("android_bootstrap")
                        .callAttr("start", srcDir().getAbsolutePath(),
                                dataDir().getAbsolutePath(), 8000)
                        .toString();
                ui(status, "brain starting (" + msg + ")…");
                waitForServer(status);
            } catch (Exception e) {
                ui(status, "⚠ failed to start brain: " + e.getMessage()
                        + "\nTap ⚙ → Download again.");
                addGear();
            }
        }).start();
    }

    private void waitForServer(TextView status) throws Exception {
        long deadline = System.currentTimeMillis() + 240_000;
        while (System.currentTimeMillis() < deadline) {
            try {
                HttpURLConnection c = (HttpURLConnection)
                        new URL("http://127.0.0.1:8000/api/status").openConnection();
                c.setConnectTimeout(1500);
                c.setReadTimeout(1500);
                if (c.getResponseCode() == 200) {
                    runOnUiThread(() -> { showBrain("http://127.0.0.1:8000"); addGear(); });
                    return;
                }
            } catch (Exception ignored) { }
            Thread.sleep(1000);
        }
        ui(status, "⚠ brain did not come up in time — tap ⚙ and retry");
        addGear();
    }

    // ------------------------------------------------------------- webview
    @SuppressLint("SetJavaScriptEnabled")
    private void showBrain(String url) {
        web = new WebView(this);
        web.getSettings().setJavaScriptEnabled(true);
        web.getSettings().setDomStorageEnabled(true);
        web.setWebViewClient(new WebViewClient());
        web.loadUrl(url);
        root.addView(web, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT));
    }

    private void addGear() {
        TextView gear = new TextView(this);
        gear.setText("⚙");
        gear.setTextSize(18);
        gear.setTextColor(FG);
        gear.setGravity(Gravity.CENTER);
        gear.setBackgroundColor(0x447C5CFF);
        gear.setPadding(dp(8), dp(4), dp(8), dp(4));
        gear.setOnClickListener(v -> { root.removeAllViews(); showSetup(); });
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(
                dp(40), dp(40), Gravity.TOP | Gravity.END);
        lp.setMargins(0, dp(8), dp(8), 0);
        root.addView(gear, lp);
    }

    private void ui(TextView t, String msg) {
        runOnUiThread(() -> t.setText(msg));
    }

    @Override
    public void onBackPressed() {
        if (web != null && web.canGoBack()) web.goBack();
        else super.onBackPressed();
    }

    @Override
    protected void onDestroy() {
        if (web != null) web.destroy();
        super.onDestroy();
    }
}
