package weka.classifiers.lazy.kstar.gpu;


import com.nativelibs4java.opencl.*;
import com.nativelibs4java.util.IOUtils;
import org.bridj.Pointer;
import weka.classifiers.lazy.ParallelKStar;
import weka.classifiers.lazy.kstar.parallel.KStarComputationResults;
import weka.classifiers.lazy.kstar.parallel.KStarComputationUnit;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.nio.ByteOrder;
import java.util.Enumeration;

import static org.bridj.Pointer.*;

public class KStarOpenCLProxy {

    //OpenCL
    static CLContext context = null;
    static CLQueue queue;
    static ByteOrder byteOrder;
    static CLProgram program = null;
    static int bufferOpenCLMode = -1;
    private final int openCLMode;

    private CLBuffer<Float> bufInstanceAttributeValues;
    private CLBuffer<Byte> bufInstanceAttributeExists;
    private CLBuffer<Integer> bufAttributeNumValues;
    private CLBuffer<Integer> bufAttributeTypes;
    private CLBuffer<Integer> bufAttributeDistribution;
    private CLBuffer<Float> bufToCompareInstanceAttributeValues;
    private CLBuffer<Byte> bufToCompareInstanceAttributeExists;
    private CLBuffer<Float> classProbability;
    private CLBuffer<Float> predictedValue;


    //Allocated memory
    int totalSize;
    Pointer<Float> instanceAttributeValues;
    Pointer<Byte> instanceAttributeExists;
    Pointer<Integer> attributeNumValues;
    Pointer<Integer> attributeTypes;
    Pointer<Integer> attributesDistribution;
    Pointer<Float> toCompareInstanceAttributeValues;
    Pointer<Byte> toCompareInstanceAttributeExists;
    private Pointer<Float> classProbabilityPtr;
    private Pointer<Float> predictedValuePtr;
    private Pointer<Float> transfProbPtr;

    //Data
    private int numberOfClasses;
    private int numberOfInstances;
    private int numberOfAttributes;
    private int largestAttributeNumValues;
    private Instance toCompare;
    private Instances computationData;

    KStarComputationUnit.KStarConfig computationConfiguration;

    private CLBuffer<Float> attrTransCacheStop;
    private CLBuffer<Float> attrTransCacheMiss;
    private CLBuffer<Byte> attrTransCacheExists;
    private CLBuffer<Float> distances;
    private CLBuffer<Float> transfProb;



    public KStarOpenCLProxy(Instance instance, Instances train, int numberOfClasses, int numberOfAttributes, int numberOfInstances, KStarComputationUnit.KStarConfig threadConfig, int openCLMode) {
        this.numberOfInstances = numberOfInstances;
        this.numberOfAttributes = numberOfAttributes;
        this.numberOfClasses = numberOfClasses;
        this.openCLMode = openCLMode;
        toCompare = instance;
        computationData = train;
        computationConfiguration = threadConfig;
        largestAttributeNumValues = defineLargestNominalNumberOfOptions();
        buildAttributesDistribution();



    }

    protected int[][] m_Distribution;
    private void buildAttributesDistribution() {
        m_Distribution = new int[largestAttributeNumValues][numberOfAttributes];
        for(int column = 0; column < numberOfAttributes; column++) {
            Attribute attribute = toCompare.attribute(column);
            if(attribute.isNominal()) {
                for (int i = 0; i < numberOfInstances; i++) {
                    Instance train = computationData.instance(i);
                    if (!train.isMissing(column)) {
                        m_Distribution[(int) train.value(column)][column]++;
                    }
                }
            }
        }
    }

    private int defineLargestNominalNumberOfOptions() {
        int largest = 0;
        for(int column = 0; column < numberOfAttributes; column++) {
            Attribute attribute = toCompare.attribute(column);
            if(attribute.isNominal()) {
                if(largest < attribute.numValues()) {
                    largest = attribute.numValues();
                }
            }
        }
        return largest;
    }

    public KStarComputationResults compute() throws Exception {
        KStarComputationResults results = new KStarComputationResults(numberOfClasses);

        long startTime = System.currentTimeMillis();

        initializeOpenClContext();
        long contextTime = System.currentTimeMillis();

        buildKernel();
        long buildingKernel = System.currentTimeMillis();

        allocateMemoryForAttributes();

        moveInstancesIntoMemory();
        long allocatingAndMovingMemory = System.currentTimeMillis();

        createBuffersInDevice();
        long allocatingInDevice = System.currentTimeMillis();

        callKernelAndWaitForResults();
        long callingKernelAndWaiting = System.currentTimeMillis();

        System.out.println("Initializing: " + (contextTime-startTime) + " | Building: " + (buildingKernel-contextTime) + "| Memory: " + (allocatingAndMovingMemory-buildingKernel) + "| Kernel:" + (callingKernelAndWaiting-allocatingAndMovingMemory));

        for(int i = 0; i < numberOfInstances; i++) {
            Instance instance = computationData.instance(i);
            results.classProbability[(int)instance.classValue()] += classProbabilityPtr.get(i);
            results.predictedValue[0] += predictedValuePtr.get(i);
            results.temp += transfProbPtr.get(i);
        }

        freeMemory();

        return results;
    }

    private void freeMemory() {
        attributeNumValues.release();
        attributeTypes.release();
        attributesDistribution.release();

        toCompareInstanceAttributeValues.release();
        toCompareInstanceAttributeExists.release();

        instanceAttributeValues.release();
        instanceAttributeExists.release();

        classProbabilityPtr.release();
        predictedValuePtr.release();
        transfProb.release();

        try {
            releaseContextData();
        } catch (Exception e) {

        }
    }

    private void releaseContextData() {
        classProbability.release();
        predictedValue.release();
        transfProb.release();

        attrTransCacheStop.release();
        attrTransCacheMiss.release();
        attrTransCacheExists.release();
        distances.release();

        bufInstanceAttributeValues.release();
        bufInstanceAttributeExists.release();
        bufAttributeNumValues.release();
        bufAttributeTypes.release();
        bufAttributeDistribution.release();
        bufToCompareInstanceAttributeValues.release();
        bufToCompareInstanceAttributeExists.release();
    }

    private void callKernelAndWaitForResults() throws IOException {
        CLKernel kernel = program.createKernel("instanceTransformationProbability");
        kernel.setArgs(numberOfInstances, numberOfAttributes, toCompare.classIndex(),
                bufToCompareInstanceAttributeValues, bufToCompareInstanceAttributeExists,
                bufInstanceAttributeValues, bufInstanceAttributeExists,
                bufAttributeNumValues, bufAttributeTypes, bufAttributeDistribution,
                classProbability, predictedValue, transfProb, distances, attrTransCacheStop, attrTransCacheMiss, attrTransCacheExists);
        //kernel.setLocalArg(14, ((Float.SIZE + Float.SIZE + Byte.SIZE) / 8) * numberOfAttributes );

        //kernel.setLocalArg(14, 4L * numberOfAttributes); // float
        //kernel.setLocalArg(15, 4L * numberOfAttributes); // float
        //kernel.setLocalArg(16, 1L * numberOfAttributes); // byte

        //int[] globalSizes = new int[] { (int)Math.ceil(numberOfInstances / 10) };
        int[] globalSizes = new int[] { numberOfInstances };
        CLEvent addEvt = kernel.enqueueNDRange(queue, globalSizes);

        classProbabilityPtr = classProbability.read(queue, addEvt);
        predictedValuePtr = predictedValue.read(queue, addEvt);
        transfProbPtr = transfProb.read(queue, addEvt);

    }

    private void buildKernel() throws IOException {

        if(program == null) {
            String src = IOUtils.readText(KStarOpenCLProxy.class.getResource("KStar.cl"));
            program = context.createProgram(src);
            program.build();
        }
    }

    private void createBuffersInDevice() {

        bufInstanceAttributeValues = context.createFloatBuffer(CLMem.Usage.Input, instanceAttributeValues);
        bufInstanceAttributeExists = context.createByteBuffer(CLMem.Usage.Input, instanceAttributeExists);
        bufAttributeNumValues = context.createIntBuffer(CLMem.Usage.Input, attributeNumValues);
        bufAttributeTypes = context.createIntBuffer(CLMem.Usage.Input, attributeTypes);
        bufAttributeDistribution = context.createIntBuffer(CLMem.Usage.Input, attributesDistribution);
        bufToCompareInstanceAttributeValues = context.createFloatBuffer(CLMem.Usage.Input, toCompareInstanceAttributeValues);
        bufToCompareInstanceAttributeExists = context.createByteBuffer(CLMem.Usage.Input, toCompareInstanceAttributeExists);

        classProbability = context.createFloatBuffer(CLMem.Usage.Output, numberOfInstances);
        predictedValue = context.createFloatBuffer(CLMem.Usage.Output, numberOfInstances);
        transfProb = context.createFloatBuffer(CLMem.Usage.Output, numberOfInstances);

        distances = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfInstances);

        attrTransCacheStop = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfAttributes);
        attrTransCacheMiss = context.createFloatBuffer(CLMem.Usage.InputOutput, numberOfAttributes);
        attrTransCacheExists = context.createByteBuffer(CLMem.Usage.InputOutput, numberOfAttributes );
    }

    private void moveInstancesIntoMemory() {
        moveAttributesDataIntoMemory();

        moveComputationDataIntoMemory();
    }

    private void moveComputationDataIntoMemory() {
        for(int row = 0; row < numberOfInstances; row++) {
            Instance computationInstance = computationData.instance(row);
            for(int column = 0; column < numberOfAttributes; column++) {
                int flatIndex = numberOfAttributes * row + column;

                instanceAttributeValues.set(flatIndex, (float) computationInstance.value(column));
                instanceAttributeExists.set(flatIndex, ( computationInstance.isMissing(column) ? (byte)0 : (byte)1 ));
            }
        }
    }

    private void moveAttributesDataIntoMemory() {
        for(int i = 0; i < numberOfAttributes; i++) {
            Attribute attribute = toCompare.attribute(i);
            attributeNumValues.set(i, attribute.numValues());
            attributeTypes.set(i, attribute.type());
            if(attribute.isNominal()) {
                moveAttributeDistributionIntoMemory(i);
            }

            toCompareInstanceAttributeValues.set(i, (float) toCompare.value(i)); // will lose precision
            toCompareInstanceAttributeExists.set(i, ( toCompare.isMissing(i) ? (byte)0 : (byte)1 ));


        }
    }

    private void moveAttributeDistributionIntoMemory(int attributeIndex) {
        for(int attributeValue = 0; attributeValue < largestAttributeNumValues; attributeValue++) {
            int flatIndex = numberOfAttributes * attributeValue + attributeIndex;
            attributesDistribution.set(flatIndex, m_Distribution[attributeValue][attributeIndex]);
        }

    }

    private void allocateMemoryForAttributes() {

        totalSize = numberOfInstances * numberOfAttributes;

        attributeNumValues = allocateInts(numberOfAttributes).order(byteOrder);
        attributeTypes = allocateInts(numberOfAttributes).order(byteOrder);
        attributesDistribution = allocateInts(numberOfAttributes * largestAttributeNumValues).order(byteOrder);

        toCompareInstanceAttributeValues = allocateFloats(numberOfAttributes).order(byteOrder);
        toCompareInstanceAttributeExists = allocateBytes(numberOfAttributes).order(byteOrder);

        instanceAttributeValues = allocateFloats(totalSize).order(byteOrder);
        instanceAttributeExists = allocateBytes(totalSize).order(byteOrder);


    }

    private void initializeOpenClContext() throws Exception {
        if(bufferOpenCLMode != openCLMode) {
            if(context != null) {
                context.release();
            }
            context = null;
            program = null;
        }
        if(context == null) {
            if(openCLMode == ParallelKStar.OPENCL_GPU) {
                context = JavaCL.createBestContext(CLPlatform.DeviceFeature.GPU); // force device
            } else if(openCLMode == ParallelKStar.OPENCL_CPU) {
                context = JavaCL.createBestContext(CLPlatform.DeviceFeature.CPU); // force device
            } else {
                throw new Exception("Unknown OpenCL mode");
            }

            queue = context.createDefaultQueue();
            byteOrder = context.getByteOrder();
            bufferOpenCLMode = openCLMode;
        }
    }
}
