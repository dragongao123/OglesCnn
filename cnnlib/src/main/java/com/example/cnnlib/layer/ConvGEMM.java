package com.example.cnnlib.layer;

import android.content.Context;
import android.support.v4.app.INotificationSideChannel;
import android.text.TextUtils;

import com.example.cnnlib.render.Render;
import com.example.cnnlib.utils.Constants;
import com.example.cnnlib.utils.DataUtils;
import com.example.cnnlib.utils.NetUtils;
import com.example.cnnlib.utils.ParamUnpacker;
import com.example.cnnlib.utils.ShaderUtils;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import static com.example.cnnlib.render.Render.initCompPro;
import static com.example.cnnlib.render.Render.initConvolutePro;
import static com.example.cnnlib.utils.Constants.S_CONV_GEMM_SHADER_HEADER;
import static com.example.cnnlib.utils.Constants.S_CONV_SHADER_HEADER;

/**
 *
 * */
public class ConvGEMM extends Layer {

    private static final String TAG = "ConvGEMM";
    private final int mKennelAmount;

    private int mNumGroupsX;

    private List<float[]> mKennels;
    private int[] mStrides;
    private int[] mKennelShape;
    private int mPad;
    private int mShaderPro;
    private int mNumGroupsZ;
    private int[] mParams;
    private NonLinear.Type mType;
    private String mKennelFilePath;

    private int mIndexBufferId;


    public ConvGEMM(Context context, Layer preLayer, int kAmount, int k_w, int k_h, int pad, int stride_w, int stride_h, NonLinear.Type type, String kennelFilePath) {
        super(context, preLayer);
        this.mKennelShape = new int[]{k_w, k_h, preLayer.getOutputShape()[2]};
        this.mKennelAmount = kAmount;
        this.mPad = pad;
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
        return (length + 2 * mPad - kennelLen) / stride + 1;
    }

    private int getLocalSizeX(int[] outputShape) {
        int outArea = outputShape[0] * outputShape[1];
        if (outArea >= 1024 * 4) {
            return 1024;
        } else {
            return (int) Math.ceil(outArea * 1.0f / 4);
        }
    }

    private int getLocalSizeZ(int xSize) {
        int maxZSize = 1024 / xSize >= 64 ? 64 : 1024 / xSize;
        int zSize = NetUtils.alignBy4(mOutputShape[2]) / 4;
        if (zSize >= maxZSize) {
            return maxZSize;
        } else {
            return zSize;
        }

    }

    private void initConv() {

        int xSize = getLocalSizeX(mOutputShape);
        mNumGroupsX = (int) Math.ceil(NetUtils.alignBy4(mOutputShape[0] * mOutputShape[1]) * 1.0f / 4 / xSize);

        int zSize = getLocalSizeZ(xSize);
        mNumGroupsZ = (int) Math.ceil(NetUtils.alignBy4(mOutputShape[2]) * 1.0f / 4 / zSize);

        String source = createShaderSource(xSize, zSize);
        mShaderPro = initCompPro(source);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createTexture();
        createConvInputDataIndex();

        if (TextUtils.isEmpty(mKennelFilePath)) {
            mKennels = createTestKennels();
        } else {
            mKennels = loadKennels();
        }

        createShaderParams();
    }

    private String createShaderSource(int xSize, int zSize) {
        String source = ShaderUtils.loadFromAssetsFile("conv_gemm_ssbo.comp", mContext.getResources());
        int kennelArea = mKennelShape[0] * mKennelShape[1];
        int kennelSize = kennelArea * NetUtils.alignBy4(mKennelShape[2]);
        return String.format(Locale.getDefault(), S_CONV_GEMM_SHADER_HEADER, NetUtils.alignBy4(mInputShape[2]) / 4, NetUtils.alignBy4(mOutputShape[2]) / 4, kennelArea, mKennelAmount, kennelSize, xSize, 1, zSize) + source;
    }

    private void createConvInputDataIndex() {
        int outArea = mOutputShape[0] * mOutputShape[1];            // int
        int kennelArea = mKennelShape[0] * mKennelShape[1];         // ivec2
        int inOffsetSize = NetUtils.alignBy4(mInputShape[2]) / 4;    // ivec2
        int outOffsetSize = NetUtils.alignBy4(mOutputShape[2]) / 4;    // ivec2
        int buuferSize = NetUtils.alignBy4(inOffsetSize * 2) + NetUtils.alignBy4(outOffsetSize * 2) + outArea * kennelArea * 2;
        mIndexBufferId = Render.initKennelBuffer(buuferSize);
        int[] convIndexes = new int[outArea * kennelArea * 2];

        for (int outX = 0; outX < mOutputShape[0]; outX++) {
            for (int outY = 0; outY < mOutputShape[1]; outY++) {
                for (int kX = 0; kX < mKennelShape[0]; kX++) {
                    for (int kY = 0; kY < mKennelShape[1]; kY++) {
                        int outIndex = outY * mOutputShape[0] + outX;
                        int kIndex = kY * mKennelShape[0] + kX;
                        int indexXOnFeatureMap = -mPad + kX + mStrides[0] * outX;
                        int indexYOnFeatureMap = -mPad + kY + mStrides[1] * outY;
                        if (isZero(indexXOnFeatureMap, indexYOnFeatureMap)) {
                            int zeroIndex = Constants.S_TEXTURE_SIZE - 1;
                            convIndexes[outIndex * kennelArea * 2 + kIndex * 2] = zeroIndex;
                            convIndexes[outIndex * kennelArea * 2 + kIndex * 2 + 1] = zeroIndex;
                        } else {
                            convIndexes[outIndex * kennelArea * 2 + kIndex * 2] = indexXOnFeatureMap;
                            convIndexes[outIndex * kennelArea * 2 + kIndex * 2 + 1] = indexYOnFeatureMap;
                        }
                    }
                }
            }
        }

        int[] inOffsetIndexes = new int[inOffsetSize * 2];
        for (int num = 0; num < inOffsetSize; num++) {
            int[] featureMapIndexInTexture = getFeatureMapIndexInTexture(num, mInputShape[0]);
            inOffsetIndexes[num * 2] = featureMapIndexInTexture[0] * mInputShape[0];
            inOffsetIndexes[num * 2 + 1] = featureMapIndexInTexture[1] * mInputShape[1];
        }

        int[] outOffsetIndexes = new int[outOffsetSize * 2];
        for (int num = 0; num < outOffsetSize; num++) {
            int[] featureMapIndexInTexture = getFeatureMapIndexInTexture(num, mOutputShape[0]);
            outOffsetIndexes[num * 2] = featureMapIndexInTexture[0] * mOutputShape[0];
            outOffsetIndexes[num * 2 + 1] = featureMapIndexInTexture[1] * mOutputShape[1];
        }

        Render.transferToBuffer(IntBuffer.wrap(inOffsetIndexes), mIndexBufferId, 0);
        Render.transferToBuffer(IntBuffer.wrap(outOffsetIndexes), mIndexBufferId, NetUtils.alignBy4(inOffsetIndexes.length));
        Render.transferToBuffer(IntBuffer.wrap(convIndexes), mIndexBufferId, NetUtils.alignBy4(inOffsetIndexes.length) + NetUtils.alignBy4(outOffsetIndexes.length));

    }

    private boolean isZero(int x, int y) {
        return (x < 0 || x >= mInputShape[0]) || (y < 0 || y >= mInputShape[1]);
    }

    private int[] getFeatureMapIndexInTexture(int num, int width) {
        int xMaxCount = Constants.S_TEXTURE_SIZE / width;
        int x, y = 0;
        if (num < xMaxCount) {
            x = num;
        } else {
            x = num % xMaxCount;
            y = num / xMaxCount;
        }
        return new int[]{x, y};
    }

    private void createShaderParams() {
        mParams = new int[14];
        mParams[0] = mKennelShape[0];
        mParams[1] = mKennelShape[1];
        mParams[2] = mKennelShape[2];
        mParams[3] = mInputShape[0];
        mParams[4] = mInputShape[1];
        mParams[5] = mInputShape[2];
        mParams[6] = mOutputShape[0];
        mParams[7] = mOutputShape[1];
        mParams[8] = mOutputShape[2];
        mParams[9] = mStrides[0];
        mParams[10] = mStrides[1];
        mParams[11] = mPad;
        mParams[12] = mType.index;
        mParams[13] = NetUtils.alignBy4(mInputShape[2]);
    }

    private List<float[]> createTestKennels() {
        List<float[]> kennels = new ArrayList<>();
        for (int i = 0; i < mOutputShape[2]; i++) {
            kennels.add(DataUtils.createConvKennel(i + 1, mKennelShape[0], mKennelShape[1], mKennelShape[2], 1));
        }
        return kennels;
    }

    /**
     * TODO
     * channel需要4对齐, 依次排满一个通道后再排下一个通道的值, 最后4个值为 bias, 0, 0, 0
     */
    private List<float[]> loadKennels() {
        return null;
    }


    @Override
    public void initialize() {
        initConv();
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureAndBuffer(mOutTex, mAttachID, mIndexBufferId);
    }

    @Override
    protected void actualForwardProc(float[][][] input) {
        Render.performConvoluteSSBO(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mIndexBufferId, mNumGroupsX, 1, mNumGroupsZ);
    }
}
