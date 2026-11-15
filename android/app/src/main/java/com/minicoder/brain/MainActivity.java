package com.minicoder.brain;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.os.Bundle;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

/**
 * MiniCoder Brain — thin-client app.
 *
 * The brain (NumPy model + memory + sleep engine) runs as a server; this app
 * is a WebView aimed at its dashboard. The brain would crawl if it ran
 * ON the phone — the phone is the sensor/screen, the server does the thinking.
 */
public class MainActivity extends Activity {

    private static final String PREFS = "brain";
    private static final String KEY_URL = "brain_url";
    private static final int BG = 0xFF0D1117, FG = 0xFFC9D1D9, ACC = 0xFF7C5CFF,
            BORDER = 0xFF30363D;

    private SharedPreferences prefs;
    private FrameLayout root;
    private WebView web;

    private int dp(int v) {
        return (int) TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, v, getResources().getDisplayMetrics());
    }

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        root = new FrameLayout(this);
        root.setBackgroundColor(BG);
        setContentView(root);
        String url = prefs.getString(KEY_URL, "");
        if (url.isEmpty()) showConfig(); else showBrain(url);
        addGear();
    }

    // ------------------------------------------------------------- config UI
    private void showConfig() {
        LinearLayout col = new LinearLayout(this);
        col.setOrientation(LinearLayout.VERTICAL);
        col.setGravity(Gravity.CENTER);
        col.setPadding(dp(32), dp(24), dp(32), dp(24));

        TextView title = new TextView(this);
        title.setText("🧠 MiniCoder Brain");
        title.setTextColor(FG);
        title.setTextSize(26);
        title.setGravity(Gravity.CENTER);
        col.addView(title);

        TextView sub = new TextView(this);
        sub.setText("Connect to your running brain dashboard\n"
                + "(start one with: minicoder dashboard --port 8000)");
        sub.setTextColor(0xFF8B949E);
        sub.setTextSize(13);
        sub.setGravity(Gravity.CENTER);
        sub.setPadding(0, dp(10), 0, dp(28));
        col.addView(sub);

        final EditText in = new EditText(this);
        in.setHint("http://192.168.1.20:8000  or  https://your-brain.example");
        in.setHintTextColor(0xFF484F58);
        in.setTextColor(FG);
        in.setSingleLine(true);
        in.setText(prefs.getString(KEY_URL, ""));
        in.setPadding(dp(12), dp(8), dp(12), dp(8));
        col.addView(in, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));

        Button go = new Button(this);
        go.setText("Connect →");
        go.setTextColor(Color.WHITE);
        go.setBackgroundColor(ACC);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.setMargins(0, dp(20), 0, 0);
        col.addView(go, lp);
        go.setOnClickListener(v -> {
            String url = in.getText().toString().trim();
            if (url.isEmpty()) return;
            if (!url.startsWith("http")) url = "http://" + url;
            prefs.edit().putString(KEY_URL, url).apply();
            root.removeAllViews();
            showBrain(url);
            addGear();
        });

        TextView tip = new TextView(this);
        tip.setText("\nTip: your computer and phone just need the same Wi-Fi — "
                + "run the dashboard on the computer, type its IP:port here.");
        tip.setTextColor(0xFF484F58);
        tip.setTextSize(11);
        tip.setGravity(Gravity.CENTER);
        col.addView(tip);

        root.addView(col, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT));
    }

    // -------------------------------------------------------------- webview
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

    /** tiny translucent ⚙ button — reopen the server settings */
    private void addGear() {
        TextView gear = new TextView(this);
        gear.setText("⚙");
        gear.setTextSize(18);
        gear.setTextColor(FG);
        gear.setGravity(Gravity.CENTER);
        gear.setBackgroundColor(0x447C5CFF);
        gear.setPadding(dp(8), dp(4), dp(8), dp(4));
        gear.setOnClickListener(v -> {
            root.removeAllViews();
            showConfig();
        });
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(
                dp(40), dp(40), Gravity.TOP | Gravity.END);
        lp.setMargins(0, dp(8), dp(8), 0);
        root.addView(gear, lp);
    }

    @Override
    public void onBackPressed() {
        if (web != null && web.canGoBack()) {
            web.goBack();
        } else {
            super.onBackPressed();
        }
    }

    @Override
    protected void onDestroy() {
        if (web != null) web.destroy();
        super.onDestroy();
    }
}
