package com.example.cnnlib.layer;

import android.content.Context;
import android.text.TextUtils;

import com.example.cnnlib.render.Render;
import com.example.cnnlib.utils.DataUtils;
import com.example.cnnlib.utils.NetUtils;
import com.example.cnnlib.utils.ParamUnpacker;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;

import static com.example.cnnlib.render.Render.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.Render.getCompShaderLocalSizeZ;
import static com.example.cnnlib.render.Render.initConvolutePro;

/**
 * 使用 texture 存储kennel
 * 每个计算器同时计算出输出的4个通道上的数据
 * 注意：输入,输出,kennel的通道都必须4对齐
 */
public class ConvTexTest extends Layer {

    private static final String TAG = "ConvTex";

    private List<float[]> mKennels;
    private int[] mStrides;
    private int[] mKennelShape;
    private int mPadding;
    private int mShaderPro;
    private int mNumGroupsY;
    private int mNumGroupsZ;
    private int[] mParams;
    private NonLinear.Type mType;
    private String mKennelFilePath;

    private int mKennelTex;
    private int mKennelAttachId;

    private int mStartY;
    private int mOperatorBuffer;


    public ConvTexTest(Context context, Layer preLayer, int kAmount, int k_w, int k_h, int pad, int stride_w, int stride_h, NonLinear.Type type, String kennelFilePath) {
        super(context, preLayer);
        this.mKennelShape = new int[]{k_w, k_h, preLayer.getOutputShape()[2]};
        this.mPadding = pad;
        this.mStrides = new int[]{stride_w, stride_h};
        this.mType = type;
        this.mOutputShape = calculateConvShape(kAmount);
        this.mKennelFilePath = kennelFilePath;
    }

    private int[] calculateConvShape(int kennelAmount) {
        int[] inShape = mPreLayer.getOutputShape();
        int width = calculateLength(inShape[0], mKennelShape[0], mStrides[0]);
        int height = calculateLength(inShape[1], mKennelShape[1], mStrides[1]);
        return new int[]{width, height, kennelAmount};
    }

    private int calculateLength(int length, int kennelLen, int stride) {
        return (length + 2 * mPadding - kennelLen) / stride + 1;
    }

    private void initConv() {
        int localSizeY = getCompShaderLocalSizeY(mOutputShape);
        mNumGroupsY = (int) Math.ceil(mOutputShape[1] * 1.0d / localSizeY);
        int localSizeZ = 1;
        mNumGroupsZ = mOutputShape[2] /4;

        mShaderPro = initConvolutePro(mContext, "conv_test.comp", mKennelShape, mOutputShape[2], mOutputShape[0], localSizeY, localSizeZ);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createTexture();

        mKennelAttachId = Layer.getConvKennelAttachID();
        mKennelTex = Render.getConvKennelTexture();

        mStartY = Layer.getConvStartY(mOutputShape[1]);
        if (TextUtils.isEmpty(mKennelFilePath)) {
            mKennels = createTestKennels();
        } else {
            mKennels = loadKennels();
        }
        transferKennelToTex();


        int[] operatorIndex = createOperatorIndex();

        int kennelBufSize = operatorIndex.length;
        mOperatorBuffer = Render.initKennelBuffer(kennelBufSize);
        transferKennelToBuffer(operatorIndex);


        mParams = new int[14];
        int[] inputShape = mPreLayer.getOutputShape();
        mParams[0] = mKennelShape[0];
        mParams[1] = mKennelShape[1];
        mParams[2] = NetUtils.alignBy4(mKennelShape[2]);
        mParams[3] = inputShape[0];
        mParams[4] = inputShape[1];
        mParams[5] = NetUtils.alignBy4(inputShape[2]);
        mParams[6] = mOutputShape[0];
        mParams[7] = mOutputShape[1];
        mParams[8] = mOutputShape[2];
        mParams[9] = mStrides[0];
        mParams[10] = mStrides[1];
        mParams[11] = mPadding;
        mParams[12] = mType.index;
        mParams[13] = mStartY;
    }

    private void transferKennelToBuffer(int[] operatorIndex) {
        Render.transferToBuffer(IntBuffer.wrap(operatorIndex), mOperatorBuffer, 0);
    }

    private int[] createOperatorIndex() {
        int[] operatorIndex = new int[mOutputShape[0] * mOutputShape[1] * mKennelShape[0] * mKennelShape[1] * 4];

        for (int out_w = 0; out_w < mOutputShape[0]; out_w++) {
            for (int out_h = 0; out_h < mOutputShape[1]; out_h++) {
                for (int k_w = 0; k_w < mKennelShape[0]; k_w++) {
                    for (int k_h = 0; k_h < mKennelShape[1]; k_h++) {
                        int current_in_index = out_w + out_h * mOutputShape[0];
                        int current_k_index = k_w + k_h * mKennelShape[0];
                        int start_index = current_in_index * 4 * mKennelShape[0] * mKennelShape[1] + 4 * current_k_index;
                        operatorIndex[start_index] = -1 * mPadding + mStrides[0] * out_w;
                        operatorIndex[start_index + 1] = -1 * mPadding + mStrides[1] * out_h;
                        operatorIndex[start_index + 2] = k_w;
                        operatorIndex[start_index + 3] = k_h;
                    }
                }
            }
        }
        return operatorIndex;
    }

    private List<float[]> createTestKennels() {
        List<float[]> kennels = new ArrayList<>();
        for (int i = 0; i < mOutputShape[2]; i++) {
            kennels.add(DataUtils.createConvKennel(i + 1, mKennelShape[0], mKennelShape[1], mKennelShape[2], 1));
        }
        return kennels;
    }

    /**
     * channel需要4对齐, 依次排满一个通道后再排下一个通道的值, 最后4个值为 bias, 0, 0, 0
     */
    private List<float[]> loadKennels() {
        ParamUnpacker paramUnpacker = new ParamUnpacker();
        Object[] objects = paramUnpacker.unpackerFunction(mKennelFilePath, new Class[]{float[][][][].class, float[].class});
        float[][][][] localWeight = (float[][][][]) objects[0];
        float[] localBias = (float[]) objects[1];

        int alignChannel = NetUtils.alignBy4(mKennelShape[2]);
        List<float[]> kennels = new ArrayList<>();
        for (int i = 0; i < mOutputShape[2]; i++) {
            float[] kennel = new float[mKennelShape[0] * mKennelShape[1] * alignChannel + 4];
            for (int c = 0; c < mKennelShape[2]; c++) {
                for (int w = 0; w < mKennelShape[0]; w++) {
                    for (int h = 0; h < mKennelShape[1]; h++) {
                        kennel[(h * mKennelShape[0] + w) * alignChannel + c] = localWeight[i][c][h][w];
                    }
                }
            }
            kennel[mKennelShape[0] * mKennelShape[1] * alignChannel] = localBias[i];
            kennels.add(kennel);
        }
        return kennels;
    }

    private void transferKennelToTex() {
        Render.bindTextureAndBuffer(mKennelTex, mKennelAttachId);
        for (int i = 0; i < mKennels.size(); i++) {
            float[] kennel = mKennels.get(i);
            int width = kennel.length / 4;
            Render.transferToTexture(FloatBuffer.wrap(kennel), mKennelTex, 0, mStartY + i, width, 1);
        }
    }

    @Override
    public void initialize() {
        initConv();
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureAndBuffer(mOutTex, mAttachID, mOperatorBuffer);

    }

    @Override
    protected void actualForwardProc(float[][][] input) {
        Render.performConvoluteTexTest(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mKennelTex, mOperatorBuffer, mNumGroupsY, mNumGroupsZ);
    }

}
