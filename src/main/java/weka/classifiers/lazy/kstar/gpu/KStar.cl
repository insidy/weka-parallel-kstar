#define BLEND_FACTOR 20
#define ROOT_FINDER_ACCURACY 0.01f
#define ROOT_FINDER_MAX_ITER 40
#define EPSILON 0.00001f

typedef struct tag_my_struct
{
    float sphere;
    float actEntropy;
    float randEntropy;
    float avgProb;
    float minProb;
} KStarWrapper;

typedef struct tag_my_struct2
{
    float stopProb;
    float missingProb;
    char isActive;
} KStarAttrCache;

void initializeWrapper(KStarWrapper* wrapper)
{
    wrapper[0].sphere = 0.0f;
    wrapper[0].actEntropy = 0.0f;
    wrapper[0].randEntropy = 0.0f;
    wrapper[0].avgProb = 0.0f;
    wrapper[0].minProb = 0.0f;
}

void calculateSphereSize(int testvalue, float stop, int numberOfAttributeValues, int numberOfAttributes, int idxAttr, int validAttributeCount, KStarWrapper* params, __global const int* attributesDistribution)
{
    int i = 0, thiscount = 0;
    float tprob = 0.0f, tval = 0.0f, t1 = 0.0f;
    float sphere = 0.0f, minprob = 1.0f, transprob = 0.0f;

    for (i = 0; i < numberOfAttributeValues; i++) {
      int flatIndexForAttrDist = numberOfAttributes * i + idxAttr;
      thiscount = attributesDistribution[flatIndexForAttrDist];
      if (thiscount != 0) {
        if (testvalue == i) {
          tprob = (stop + (1 - stop) / numberOfAttributeValues) / validAttributeCount;
          tval += tprob * thiscount;
          t1 += tprob * tprob * thiscount;
        } else {
          tprob = ((1 - stop) / numberOfAttributeValues) / validAttributeCount;
          tval += tprob * thiscount;
          t1 += tprob * tprob * thiscount;
        }
        if (minprob > tprob * validAttributeCount) {
          minprob = tprob * validAttributeCount;
        }
      }
    }
    transprob = tval;
    if(t1 == 0) {
        sphere = 0;
    } else {
        sphere = ((tval * tval) / t1);
    }

    // return values ... Yck!!!
    params[0].sphere = sphere;
    params[0].avgProb = transprob;
    params[0].minProb = minprob;
}

void stopProbUsingBlend(float* stopProb, float* missingProb, __global const int* attributesDistribution, float toCompareAttributeValue, int validAttributeCount, int numberOfAttributeValues, int numberOfAttributes, int idxAttr )
{
    // Math.abs = fabs
    int itcount = 0;
    KStarWrapper botvals, upvals, vals;
    initializeWrapper(&botvals);
    initializeWrapper(&upvals);
    initializeWrapper(&vals);

    int testvalue = toCompareAttributeValue;
    int flatIndexForAttrDist = numberOfAttributes * testvalue + idxAttr;
    float aimfor = ( validAttributeCount - attributesDistribution[flatIndexForAttrDist] ) * BLEND_FACTOR / 100.0f + attributesDistribution[flatIndexForAttrDist];
    float tstop = 1.0f - BLEND_FACTOR / 100.0f;
    float lower = 0.0f + ROOT_FINDER_ACCURACY / 2.0f;
    float upper = 1.0f - ROOT_FINDER_ACCURACY / 2.0f;

    calculateSphereSize(testvalue, lower, numberOfAttributeValues, numberOfAttributes, idxAttr, validAttributeCount, &botvals, attributesDistribution);
    botvals.sphere -= aimfor;
    calculateSphereSize(testvalue, upper, numberOfAttributeValues, numberOfAttributes, idxAttr, validAttributeCount, &upvals, attributesDistribution);
    upvals.sphere -= aimfor;

    if (upvals.avgProb == 0) {
      // When there are no training instances with the test value:
      // doesn't matter what exact value we use for tstop, just acts as
      // a constant scale factor in this case.
      calculateSphereSize(testvalue, tstop, numberOfAttributeValues, numberOfAttributes, idxAttr, validAttributeCount, &vals, attributesDistribution);
    } else if (upvals.sphere > 0) {
      // Can't include aimfor instances, going for min possible
      tstop = upper;
      vals.avgProb = upvals.avgProb;
    } else {
      // Enter the root finder
      for (;;) {
        itcount++;
        calculateSphereSize(testvalue, tstop, numberOfAttributeValues, numberOfAttributes, idxAttr, validAttributeCount, &vals, attributesDistribution);
        vals.sphere -= aimfor;
        if (fabs(vals.sphere) <= ROOT_FINDER_ACCURACY || itcount >= ROOT_FINDER_MAX_ITER) {
          break;
        }
        if (vals.sphere > 0.0f) {
          lower = tstop;
          tstop = (upper + lower) / 2.0f;
        } else {
          upper = tstop;
          tstop = (upper + lower) / 2.0f;
        }
      }
    }

    missingProb[0] = vals.avgProb;

    /*
    m_SmallestProb = vals.minProb;
    m_AverageProb = vals.avgProb;
    // Set the probability of transforming to a missing value
    switch (m_MissingMode) {
    case M_DELETE:
      m_MissingProb = 0.0;
      break;
    case M_NORMAL:
      m_MissingProb = 1.0;
      break;
    case M_MAXDIFF:
      m_MissingProb = m_SmallestProb;
      break;
    case M_AVERAGE:
      m_MissingProb = m_AverageProb;
      break;
    }*/

    if (fabs(vals.avgProb - validAttributeCount) < EPSILON) {
      // No difference in the values
      stopProb[0] = 1.0f;
    } else {
      stopProb[0] = tstop;
    }

}

int getAttributeCount(int idxAttr, int numberOfInstances, int numberOfAttributes, __global const char* instanceAttributeExists)
{
    int totalCount = 0;
    int idxInstance = 0;
    for(idxInstance = 0; idxInstance < numberOfInstances; idxInstance++)
    {
        int flatIndex = ( numberOfAttributes * idxInstance + idxAttr );
        if(instanceAttributeExists[idxInstance] == 1)
        {
            totalCount++;
        }
    }

    return totalCount;
}
float nominalAttrTransProb(int idxAttr, int currentThread, int numberOfInstances, int numberOfAttributes,
                    __global const float* toCompareInstanceAttributeValues, __global const char* toCompareInstanceAttributeExists,
                    __global const float* instanceAttributeValues, __global const char* instanceAttributeExists,
                    __global const int* attributeNumValues, __global const int* attributesDistribution
                    //,__local KStarAttrCache* attrTransCache
                    )

{
    float transProb = 0.0f;
    float stopProb = 0.0f;
    float missingProb = 1.0f;
    int validAttributeCount = getAttributeCount(idxAttr, numberOfInstances, numberOfAttributes, instanceAttributeExists);
    float toCompareAttributeValue = toCompareInstanceAttributeValues[idxAttr];

    /*
    int flatIdxForCache = idxAttr;
    KStarAttrCache existingCache = attrTransCache[ flatIdxForCache ];
    if(existingCache.isActive == 1) {
        stopProb = existingCache.stopProb;
        missingProb = existingCache.missingProb;
    } else {
        stopProbUsingBlend(&stopProb, &missingProb, attributesDistribution, toCompareAttributeValue, validAttributeCount, attributeNumValues[idxAttr], numberOfAttributes, idxAttr);
        KStarAttrCache newCache;
        newCache.stopProb = stopProb;
        newCache.missingProb = missingProb;
        newCache.isActive = 1;
        attrTransCache[ flatIdxForCache ] = newCache;
    }*/

    stopProbUsingBlend(&stopProb, &missingProb, attributesDistribution, toCompareAttributeValue, validAttributeCount, attributeNumValues[idxAttr], numberOfAttributes, idxAttr);

    int flatIndex = ( numberOfAttributes * currentThread + idxAttr );
    if(instanceAttributeExists[flatIndex] == 0) {
        transProb = missingProb;
    } else {
        transProb = (1.0f - stopProb) / attributeNumValues[idxAttr];
        if ((int) toCompareInstanceAttributeValues[idxAttr] == (int) instanceAttributeValues[flatIndex]) {
          transProb += stopProb;
        }
    }
    return transProb;
}

float PStar(float x, float scale) {
    return scale * exp(-2.0 * x * scale);
}

void calculateSphereSizeNumeric(int numberOfInstances, int actualCount, __global float* distances, float scale, KStarWrapper* params) {
    int i;
    float sphereSize, minprob = 1.0f;
    float pstar; // P*(b|a)
    float pstarSum = 0.0; // sum(P*)
    float pstarSquareSum = 0.0; // sum(P*^2)
    float inc;
    for (i = 0; i < numberOfInstances; i++) {
      if (distances[i] < 0) {
        // instance with missing value
        continue;
      } else {
        pstar = PStar(distances[i], scale);
        if (minprob > pstar) {
          minprob = pstar;
        }
        inc = pstar / actualCount;
        pstarSum += inc;
        pstarSquareSum += inc * inc;
      }
    }
    if(pstarSquareSum == 0) {
        sphereSize = 0;
    } else {
        sphereSize = pstarSum * pstarSum / pstarSquareSum;
    }

    // return the values
    params[0].sphere = sphereSize;
    params[0].avgProb = pstarSum;
    params[0].minProb = minprob;

}

void scaleFactorUsingBlend( float* scale, float* missingProb, int* actualCount,
                            int idxAttr, int currentThread, int numberOfInstances, int numberOfAttributes,
                            __global const float* toCompareInstanceAttributeValues, __global const char* toCompareInstanceAttributeExists,
                            __global const float* instanceAttributeValues, __global const char* instanceAttributeExists,
                            __global float* distances) {
    KStarWrapper botvals, upvals, vals;
    initializeWrapper(&botvals);
    initializeWrapper(&upvals);
    initializeWrapper(&vals);

    int i, j, lowestcount = 0;
    float lowest = -1.0f, nextlowest = -1.0f;
    float root, broot, up, bot;
    float aimfor, min_val = 9.0e30f;
    float avgprob = 0.0f, minprob = 0.0f, min_pos = 0.0f;

    scale[0] = 1.0f;

    int flatIndex = ( numberOfAttributes * currentThread + idxAttr );
    for (j = 0; j < numberOfInstances; j++) {
        int flatInstanceIndex = ( numberOfAttributes * j + idxAttr );
        if (instanceAttributeExists[flatInstanceIndex] == 0) {
        // mark the train instance with a missing value by setting
        // the distance to -1.0
            distances[j] = -1.0f;
        } else {
            distances[j] = fabs(instanceAttributeValues[flatInstanceIndex] - toCompareInstanceAttributeValues[idxAttr]); // iteration instance - test instance values
            if ((distances[j] + EPSILON) < nextlowest || nextlowest == -1.0f) {
              if ((distances[j] + EPSILON) < lowest || lowest == -1.0f) {
                nextlowest = lowest;
                lowest = distances[j];
                lowestcount = 1;
              } else if (fabs(distances[j] - lowest) < EPSILON) {
                // record the number training instances (number n0) at
                // the smallest distance from test instance
                lowestcount++;
              } else {
                nextlowest = distances[j];
              }
            }
            actualCount[0]++;
        }
    }


    if (nextlowest == -1 || lowest == -1) { // Data values are all the same
      scale[0] = 1.0f;
      missingProb[0] = 1.0f;
      ////m_SmallestProb = m_AverageProb = 1.0; // guess that will need this...
      return;
    } else {
        // starting point for root
        root = 1.0f / (nextlowest - lowest);
        i = 0;
        // given the expression: n0 <= E(scale) <= N
        // E(scale) = (N - n0) * b + n0 with blending factor: 0 <= b <= 1
        // aimfor = (N - n0) * b + n0
        aimfor = (actualCount[0] - lowestcount) * (float) BLEND_FACTOR / 100.0f + lowestcount;
        if (BLEND_FACTOR == 0) {
            aimfor += 1.0f;
        }

        // root is bracketed in interval [bot,up]
        bot = 0.0f + ROOT_FINDER_ACCURACY / 2.0f;
        up = root * 16; // This is bodgy
        // E(bot)
        calculateSphereSizeNumeric(numberOfInstances, actualCount[0], distances, bot, &botvals);
        botvals.sphere -= aimfor;
        // E(up)
        calculateSphereSizeNumeric(numberOfInstances, actualCount[0], distances, up, &upvals);
        upvals.sphere -= aimfor;

        if (botvals.sphere < 0) { // Couldn't include that many
            // instances - going for max possible
            min_pos = bot;
            avgprob = botvals.avgProb;
            minprob = botvals.minProb;
        } else if (upvals.sphere > 0) { // Couldn't include that few,
            // going for min possible
            min_pos = up;
            avgprob = upvals.avgProb;
            minprob = upvals.minProb;
        } else {
            // Root finding Algorithm starts here !
            for (;;) {
                calculateSphereSizeNumeric(numberOfInstances, actualCount[0], distances, root, &vals);
                vals.sphere -= aimfor;
                if (fabs(vals.sphere) < min_val) {
                    min_val = fabs(vals.sphere);
                    min_pos = root;
                    avgprob = vals.avgProb;
                    minprob = vals.minProb;
                }
                if (fabs(vals.sphere) <= ROOT_FINDER_ACCURACY) {
                    break; // converged to a solution, done!
                }
                if (vals.sphere > 0.0f) {
                    broot = (root + up) / 2.0f;
                    bot = root;
                    root = broot;
                } else {
                    broot = (root + bot) / 2.0f;
                    up = root;
                    root = broot;
                }
                i++;
                if (i > ROOT_FINDER_MAX_ITER) {
                    root = min_pos;
                    break;
                }
            }
        }

        missingProb[0] = avgprob;
        scale[0] = min_pos;
        return;
    }

}

float numericAttrTransProb(int idxAttr, int currentThread, int numberOfInstances, int numberOfAttributes,
                    __global const float* toCompareInstanceAttributeValues, __global const char* toCompareInstanceAttributeExists,
                    __global const float* instanceAttributeValues, __global const char* instanceAttributeExists,
                    __global const int* attributeNumValues, __global const int* attributesDistribution,
                    __global float* distances
                    ,__global float* attrTransCacheStop
                    ,__global float* attrTransCacheMiss
                    ,__global char* attrTransCacheExists
                    )
{
    float transProb = 0.0f;
    float scale = 0.0f;
    float distance = 0.0f;
    float missingProb = 1.0f;
    int actualCount = 0;

    int flatIndex = ( numberOfAttributes * currentThread + idxAttr );

    int flatIdxForCache = idxAttr;

    if(attrTransCacheExists[ flatIdxForCache ] == 1) {
        scale = attrTransCacheStop[ flatIdxForCache ];
        missingProb = attrTransCacheMiss[ flatIdxForCache ];
    } else {
        scaleFactorUsingBlend(&scale, &missingProb, &actualCount, idxAttr, currentThread, numberOfInstances, numberOfAttributes, toCompareInstanceAttributeValues, toCompareInstanceAttributeExists, instanceAttributeValues, instanceAttributeExists, distances);

        attrTransCacheStop[ flatIdxForCache ] = scale;
        attrTransCacheMiss[ flatIdxForCache ] = missingProb;
        attrTransCacheExists[ flatIdxForCache ] = 1;
    }
    //scaleFactorUsingBlend(&scale, &missingProb, &actualCount, idxAttr, currentThread, numberOfInstances, numberOfAttributes, toCompareInstanceAttributeValues, toCompareInstanceAttributeExists, instanceAttributeValues, instanceAttributeExists, distances);



    if(instanceAttributeExists[flatIndex] == 0) {
        transProb = missingProb;
    } else {
        float toCompareAttributeValue = toCompareInstanceAttributeValues[idxAttr];
        float currentInstanceAttributeValue = instanceAttributeValues[flatIndex];

        distance = fabs(toCompareAttributeValue - currentInstanceAttributeValue);
        transProb = PStar(distance, scale);
    }
    return transProb;
}

float attrTransProb(int idxAttr, int currentThread, int numberOfInstances, int numberOfAttributes,
                    __global const float* toCompareInstanceAttributeValues, __global const char* toCompareInstanceAttributeExists,
                    __global const float* instanceAttributeValues, __global const char* instanceAttributeExists,
                    __global const int* attributeNumValues, __global const int* attributeTypes, __global const int* attributesDistribution
                    ,__global float* distances
                    ,__global float* attrTransCacheStop
                    ,__global float* attrTransCacheMiss
                    ,__global char* attrTransCacheExists
                    )
{

    // find out column type
    float attrTransProb = 0.0f;

    if(attributeTypes[idxAttr] == 1 ) { // Nominal
        attrTransProb = nominalAttrTransProb(idxAttr, currentThread, numberOfInstances, numberOfAttributes, toCompareInstanceAttributeValues, toCompareInstanceAttributeExists, instanceAttributeValues, instanceAttributeExists, attributeNumValues, attributesDistribution /*, attrTransCache*/ );

    } else if(attributeTypes[idxAttr] == 0) { // Numeric
        attrTransProb = numericAttrTransProb(idxAttr, currentThread, numberOfInstances, numberOfAttributes, toCompareInstanceAttributeValues, toCompareInstanceAttributeExists, instanceAttributeValues, instanceAttributeExists, attributeNumValues, attributesDistribution, distances , attrTransCacheStop, attrTransCacheMiss, attrTransCacheExists );
    }

    return attrTransProb;

}

__kernel void instanceTransformationProbability(int numberOfInstances, int numberOfAttributes, int classAttributeIndex,
                                                __global const float* toCompareInstanceAttributeValues, __global const char* toCompareInstanceAttributeExists,
                                                __global const float* instanceAttributeValues, __global const char* instanceAttributeExists,
                                                __global const int* attributeNumValues, __global const int* attributeTypes, __global const int* attributesDistribution,
                                                __global float* classProbability, __global float* predictedValue, __global float* transfProb
                                                ,__global float* distances
                                                ,__global float* attrTransCacheStop
                                                ,__global float* attrTransCacheMiss
                                                ,__global char* attrTransCacheExists
                                                )
{

    int currentThread = get_global_id(0);

    if (currentThread >= numberOfInstances)
        return;

    float lfClassProbability = 0.0f;
    float lfPredictedValue = 0.0f;
    float lfTransfProb = 0.0f;

    // determine transformation probability;
    lfTransfProb = 1.0f;
    int numberOfMissingAttributes = 0;
    int idxAttr = 0;
    for(idxAttr = 0; idxAttr < numberOfAttributes; idxAttr++) {

        if(toCompareInstanceAttributeExists[idxAttr] == 0) {
            numberOfMissingAttributes++;
            continue;
        }
        //int flatIndex = ( numberOfAttributes * currentThread + idxAttr );

        lfTransfProb *= attrTransProb(idxAttr, currentThread, numberOfInstances, numberOfAttributes, toCompareInstanceAttributeValues, toCompareInstanceAttributeExists, instanceAttributeValues, instanceAttributeExists, attributeNumValues, attributeTypes, attributesDistribution, distances , attrTransCacheStop, attrTransCacheMiss, attrTransCacheExists  );

        if(numberOfMissingAttributes != numberOfAttributes)
        {
            lfTransfProb = pow(lfTransfProb, (float)numberOfAttributes / (numberOfAttributes - numberOfMissingAttributes));
        }
        else
        {
            lfTransfProb = 0.0f;
        }
    }

    lfTransfProb = lfTransfProb / numberOfInstances;

    lfClassProbability = lfTransfProb; // class probability for nominal attributes

    int currentInstanceClassAttrIndex = ( numberOfAttributes * currentThread + classAttributeIndex);
    lfPredictedValue = lfTransfProb * instanceAttributeValues[currentInstanceClassAttrIndex]; // class probability for numeric attributes

    classProbability[currentThread] = lfClassProbability; // not reduced
    predictedValue[currentThread] = lfPredictedValue;
    transfProb[currentThread] = lfTransfProb;

    barrier(CLK_GLOBAL_MEM_FENCE);

}