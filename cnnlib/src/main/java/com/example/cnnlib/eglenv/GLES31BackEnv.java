package com.example.cnnlib.eglenv;

import android.opengl.GLES31;
import android.os.Handler;
import android.os.HandlerThread;

import com.example.cnnlib.render.Render;

import static android.opengl.GLES31.GL_FRAMEBUFFER;
import static android.opengl.GLES31.glBindFramebuffer;
import static android.opengl.GLES31.glGenFramebuffers;

public class GLES31BackEnv {

    private final HandlerThread mGLThread;
    private final Handler mHandler;
    private int mWidth;
    private int mHeight;
    private EGLHelper mEGLHelper;

    private int[] fFrame = new int[1];

    final static String TAG = "GLES31BackEnv";

    public GLES31BackEnv(final int width, final int height) {
        this.mWidth = width;
        this.mHeight = height;
        mGLThread = new HandlerThread("GLThread");
        mGLThread.start();

        mHandler = new Handler(mGLThread.getLooper());
        mHandler.post(() -> {
            mEGLHelper = new EGLHelper();
            mEGLHelper.eglInit(width, height);
            initFBO();
            int maxCompSharedSize = Render.getMaxCompSharedSize();
            Render.getMaxDrawBuffers();

        });
    }

    private void initFBO() {
        glGenFramebuffers(1, fFrame, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, fFrame[0]);
    }

    public void post(Runnable runnable) {
        mHandler.post(runnable);
    }

    public void destroy() {
        mHandler.post(mEGLHelper::destroy);
    }


}
