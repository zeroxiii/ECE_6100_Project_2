// Rahmaan Lodhia
// Part 1: Branch Prediction Project
// Perceptron Implementation
// Author: Hongliang Gao;   Created: Jan 27 2011
// Description: sample predictors for cbp3.

#include <stdio.h>
#include <cassert>
#include <string.h>
#include <inttypes.h>
#include <cmath>

using namespace std;
#include "cbp3_def.h"
#include "cbp3_framework.h"

// this file includes two sample predictors
// one is a 64 KB gshare conditional predictor
// the other is a 64 KB indirect predictor indexed by (pc ^ history)

// rewind_marked is also used to show how to rewind the reader for multiple runs
// the predictor will use gshare in the first run
// the indirect predictor will be used in the second run

// NOTE: rewind_marked is only provided to help tuning work. the final
// submitted code should only include one run.


#define GSHARE_SIZE 18 // 256K 2-bit counters = 64 KB cost
#define IND_SIZE 14    // 16K 32-bit targets  = 64 KB cost
#define PERC_SIZE 7    // Perceptron entry table with 8-bit weight counters
#define GHISTORY_SIZE 64 // Global history register 
#define TRAIN_THRESHOLD 0 // Perceptron training threshold

// predictor tables
int8_t   *gtable;
uint32_t *indtable;
int8_t   percTable[1 << PERC_SIZE][GHISTORY_SIZE];

// two branch history registers:
// the framework provids real branch results at fetch stage to simplify branch history
// update for predicting later branches. however, they are not available until execution stage
// in a real machine. therefore, you can only use them to update predictors at or after the
// branch is executed.
// in this sample code, we update predictors at retire stage where uops are processed
// in order to enable easy regneration of branch history.

// cost: depending on predictor size
uint32_t brh_fetch;
uint32_t brh_retire;
uint64_t ghr_fetch;
uint64_t ghr_retire;

// Perceptron output
int32_t out;
int64_t ind;

// count number of runs
uint32_t runs;


void PredictorInit() 
{
    runs = 0;
    gtable = new int8_t[1 << GSHARE_SIZE];
    indtable = new uint32_t[1 << IND_SIZE];

    // Initialize Perceptron Table
    for (int i = 0; i < (1 << PERC_SIZE); i++)
    {
        for (int j = 0; j < GHISTORY_SIZE; j++)
            percTable[i][j] = 0;
    }

    // Initialize Global History
    ghr_fetch = 0;
    ghr_retire = 0;

    assert(gtable && indtable);
}

void PredictorReset() 
{
    // this function is called before EVERY run
    // it is used to reset predictors and change configurations

    if (runs == 0)
        printf("Predictor:gshare\nconfig: %i counters, %i KB cost\n", 1 << GSHARE_SIZE, (1 << GSHARE_SIZE) * 2 / 8 / 1024);
    else if (runs == 1)
        printf("Predictor:ind\nconfig: %i targets,  %i KB cost\n", 1 << IND_SIZE, (1 << IND_SIZE) * 4 / 1024);
    else
        printf("Predictor:perceptron\nconfig: %i entries, %i-bit global history, %i KB cost\n", 1 << PERC_SIZE, GHISTORY_SIZE, (1 << PERC_SIZE) * GHISTORY_SIZE / 1024);

    for (int i = 0; i < (1 << GSHARE_SIZE); i ++)
        gtable[i] = 0;
    for (int i = 0; i < (1 << IND_SIZE); i ++)
        indtable[i] = 0;

    for (int i = 0; i < (1 << PERC_SIZE); i++)
    {
        for (int j = 0; j < GHISTORY_SIZE; j++)
            percTable[i][j] = 0;
    }

    ghr_fetch = 0;
    ghr_retire = 0;

    brh_fetch = 0;
    brh_retire = 0;
}

void PredictorRunACycle() 
{
    // get info about what uops are processed at each pipeline stage
    const cbp3_cycle_activity_t *cycle_info = get_cycle_info();

    // make prediction at fetch stage
    for (int i = 0; i < cycle_info->num_fetch; i++) 
    {
        uint32_t fe_ptr = cycle_info->fetch_q[i];
        const cbp3_uop_dynamic_t *uop = &fetch_entry(fe_ptr)->uop;

        if (runs == 0 && uop->type & IS_BR_CONDITIONAL) 
        {
            // get prediction
            uint32_t gidx = (brh_fetch ^ uop->pc) & ((1 << GSHARE_SIZE) - 1);
            bool gpred = (gtable[gidx] >= 0);

            // report prediction:
            // you need to provide direction predictions for conditional branches,
            // targets of conditional branches are available at fetch stage.
            // for indirect branches, you need to provide target predictions.
            // you can report multiple predictions for the same branch
            // the framework will use the last reported prediction to calculate 
            // misprediction penalty
            assert(report_pred(fe_ptr, false, gpred));
        }
        else if (runs == 1 && uop->type & IS_BR_INDIRECT) 
        {
            uint32_t gidx = (brh_fetch ^ uop->pc) & ((1 << IND_SIZE) - 1);
            uint32_t gpred = indtable[gidx];

            assert(report_pred(fe_ptr, false, gpred));
        }
        else if (runs == 2 && uop->type & IS_BR_CONDITIONAL)
        {
            //Get index of perceptron table
            ind = (uop->pc) % (1 << PERC_SIZE);

            //Calculate out
            out = percTable[ind][0];
            for (int j = 1; j < GHISTORY_SIZE; j++)
            {
                if (((ghr_fetch >> (GHISTORY_SIZE - j - 1)) & 1))
                {
                    out += percTable[ind][j];
                }
                else
                {
                    out -= percTable[ind][j];
                }
            }

            //Make prediction
            bool gpred = (out >= 0);

            assert(report_pred(fe_ptr, false, gpred));
        }

        // update fetch branch history
        if (uop->type & IS_BR_CONDITIONAL)
        {
            brh_fetch = (brh_fetch << 1) | (uop->br_taken ? 1 : 0);
            ghr_fetch = (ghr_fetch << 1) | (uop->br_taken ? 1 : 0);
        }
        else if (uop_is_branch(uop->type))
        {
            brh_fetch = (brh_fetch << 1) | 1;
            ghr_fetch = (ghr_fetch << 1) | 1;
        }

    }

    for (int i = 0; i < cycle_info->num_retire; i++) 
    {
        uint32_t rob_ptr = cycle_info->retire_q[i];
        const cbp3_uop_dynamic_t *uop = &rob_entry(rob_ptr)->uop;

        if (runs == 0 && uop->type & IS_BR_CONDITIONAL) 
        {
            uint32_t gidx = (brh_retire ^ uop->pc) & ((1 << GSHARE_SIZE) - 1);

            // update predictor
            bool t = uop->br_taken;
            if (t && gtable[gidx] < 1)
                gtable[gidx] ++;
            else if (!t && gtable[gidx] > -2)
                gtable[gidx] --;
        }
        else if (runs == 1 && uop->type & IS_BR_INDIRECT) 
        {
            uint32_t gidx = (brh_retire ^ uop->pc) & ((1 << IND_SIZE) - 1);
            indtable[gidx] = uop->br_target;
        }
        else if (runs == 2 && uop->type & IS_BR_CONDITIONAL)
        {
            //Get index of perceptron table
            ind = (uop->pc) % (1 << PERC_SIZE);

            //Get t
            bool t = uop->br_taken;

            //Calculate out
            out = percTable[ind][0];
            for (int j = 1; j < GHISTORY_SIZE; j++)
            {
                if (((ghr_retire >> (GHISTORY_SIZE - j - 1)) & 1))
                {
                    out += percTable[ind][j];
                }
                else
                {
                    out -= percTable[ind][j];
                }
            }


            //Train new perceptron
            if ((abs(out) <= TRAIN_THRESHOLD) || ((out >= 0) != t))
            {
                //Check x0
                if (t == true)
                {
                    percTable[ind][0] = percTable[ind][0] + 1;
                }
                else
                {
                    percTable[ind][0] = percTable[ind][0] - 1;
                }

                //Check rest of GHR
                for(int j = 1; j < GHISTORY_SIZE; j++)
                {
                    bool x = (((ghr_retire >> (GHISTORY_SIZE - j - 1)) & 1));
                    if (t == x)
                        percTable[ind][j] = percTable[ind][j] + 1;
                    else
                        percTable[ind][j] = percTable[ind][j] - 1;
                }
            }
        }

        // update retire branch history
        if (uop->type & IS_BR_CONDITIONAL)
        {
            brh_retire = (brh_retire << 1) | (uop->br_taken ? 1 : 0);
            ghr_retire = (ghr_retire << 1) | (uop->br_taken ? 1 : 0);
        }
        else if (uop_is_branch(uop->type))
        {
            brh_retire = (brh_retire << 1) | 1;
            ghr_retire = (ghr_retire << 1) | 1;
        }
    }
}

void PredictorRunEnd() 
{
    runs ++;
    if (runs < 3) // set rewind_marked to indicate that we want more runs
        rewind_marked = true;
}

void PredictorExit() 
{
    delete [] gtable;
    delete [] indtable;
}
