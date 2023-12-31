/*
Copyright 2020 EEMBC and The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file reflects a modified version of th_lib from EEMBC. The reporting logic
in th_results is copied from the original in EEMBC.
==============================================================================*/
/// \file
/// \brief C++ implementations of submitter_implemented.h

#include "submitter_implemented.h"
#include "st_port.h"

extern uint8_t gp_buff[MAX_DB_INPUT_SIZE];
extern size_t g_buff_size;
extern size_t g_buff_pos;

int8_t aiInData_int[AI_NETWORK_IN_1_SIZE];
int8_t aiOutData[AI_NETWORK_OUT_1_SIZE];


// Implement this method to prepare for inference and preprocess inputs.
void th_load_tensor(void)
{
	for (int i = 0; i < AI_NETWORK_IN_1_SIZE; i++)
	{
		aiInData_int[i] = (int8_t)(gp_buff[i]- 128);
	}
}

// Add to this method to return real inference results.
void th_results(void)
{
	    char char_res[10];
	    const int nresults = 3;
	    /**
	     * The results need to be printed back in exactly this format; if easier
	     * to just modify this loop than copy to results[] above, do that.
	     */
	    th_printf("m-results-[");
	    int kCategoryCount = 2;
	    for (int8_t i = 0; i < kCategoryCount; i++)
	    {
	      float converted = (aiOutData[i] - (-128)) * 0.00390625;
	      snprintf(char_res, 10, "%0.3f",converted);
	      // Some platforms don't implement floating point formatting.
	      th_printf("%0.3f", converted);
	      if (i < (nresults - 2)) {
	        th_printf(",");
	      }
	    }
	    th_printf("]\r\n");
}

// Implement this method with the logic to perform one inference cycle.
void th_infer(void)
{
	AI_Run(aiInData_int, aiOutData);
}

/// \brief optional API.
void th_final_initialize(void)
{
	rxInit();
	AI_Init();
	__HAL_RCC_PWR_CLK_ENABLE();
	HAL_PWREx_EnableFastSoftStart();
	/*Enable ultra low power mode NJ: to enable with cut2.1 fix*/
	//HAL_PWREx_EnableUltraLowPowerMode();
	/*The SMPS regulator supplies the Vcore Power Domains.*/
	HAL_PWREx_ConfigSupply(PWR_SMPS_SUPPLY);
	/* Enable Power Control clock */
	__HAL_RCC_PWR_CLK_DISABLE();
	/*Disable DCACHE*/
	__HAL_RCC_DCACHE1_CLK_DISABLE();
	__HAL_RCC_MSIBIAS_SELECTION_ENABLE();

}
void th_pre() {

	HAL_NVIC_DisableIRQ(SysTick_IRQn);
	__HAL_RCC_RNG_CLK_DISABLE();
	__HAL_RCC_GPIOA_CLK_DISABLE();
	__HAL_RCC_GPIOB_CLK_DISABLE();
	__HAL_RCC_GPIOC_CLK_DISABLE();
	__HAL_RCC_GPIOD_CLK_DISABLE();
	__HAL_RCC_GPIOE_CLK_DISABLE();
	__HAL_RCC_GPIOG_CLK_DISABLE();
	__HAL_RCC_GPIOH_CLK_DISABLE();
	__HAL_RCC_GPIOI_CLK_DISABLE();

	__HAL_RCC_AHB22_CLK_DISABLE();
	__HAL_RCC_APB2_CLK_DISABLE();
	__HAL_RCC_AHB3_CLK_DISABLE();
	__HAL_RCC_APB3_CLK_DISABLE();
	__HAL_RCC_AHB21_CLK_DISABLE();

#if EE_CFG_ENERGY_MODE == 1
	__HAL_RCC_GPIOF_CLK_DISABLE();
#endif

	/*
	* SRAM Power Down in Run mode Config
	*/

	HAL_PWREx_DisableRAMsContentRunRetention(PWR_SRAM1_FULL_RUN_RETENTION);
	HAL_PWREx_EnableRAMsContentRunRetention(PWR_SRAM2_FULL_RUN_RETENTION);
	HAL_PWREx_DisableRAMsContentRunRetention(PWR_SRAM3_FULL_RUN_RETENTION);
	HAL_PWREx_DisableRAMsContentRunRetention(PWR_SRAM4_FULL_RUN_RETENTION);
}

void th_post() {

	  RCC->CFGR2 &= 0xFFE0FFFF; // AHB1/APB1 + AHB2/APB2
	  RCC->CFGR3 &= 0xFFFCFFFF; // AHB3/APB
	 __HAL_RCC_GPIOF_CLK_ENABLE();
	 HAL_NVIC_EnableIRQ(SysTick_IRQn);
}


char g_cmd_buf_copy[EE_CMD_SIZE + 1];
void th_command_ready(char *p_command)
{
	if (p_command[0])
	{
		th_memcpy(g_cmd_buf_copy, p_command, EE_CMD_SIZE);
		ee_serial_command_parser_callback(g_cmd_buf_copy);
		p_command[0] = 0;
	}
}

// th_libc implementations.
int th_strncmp(const char *str1, const char *str2, size_t n) {
  return strncmp(str1, str2, n);
}

char *th_strncpy(char *dest, const char *src, size_t n) {
  return strncpy(dest, src, n);
}

size_t th_strnlen(const char *str, size_t maxlen) {
  return strnlen(str, maxlen);
}

char *th_strcat(char *dest, const char *src) { return strcat(dest, src); }

char *th_strtok(char *str1, const char *sep) { return strtok(str1, sep); }

int th_atoi(const char *str) { return atoi(str); }

void *th_memset(void *b, int c, size_t len) { return memset(b, c, len); }

void *th_memcpy(void *dst, const void *src, size_t n) {
  return memcpy(dst, src, n);
}

/* N.B.: Many embedded *printf SDKs do not support all format specifiers. */
int th_vprintf(const char *format, va_list ap) { return vprintf(format, ap); }

uint8_t buffer [128];
void th_printf(const char* fmt, ...)
{
	va_list ap;
	int n;
	va_start(ap, fmt);
	n = vsnprintf ((char*)buffer, 128, fmt, ap);
	va_end(ap);
	//notify_uart(buffer, n);
#if EE_CFG_ENERGY_MODE == 0
	if(HAL_UART_Transmit(&huart1, (uint8_t*)buffer, n, HAL_MAX_DELAY) != HAL_OK)
	{
		Error_Handler();
	}
#else
	if(HAL_UART_Transmit(&hlpuart1, (uint8_t*)buffer, n, HAL_MAX_DELAY) != HAL_OK)
	{
		Error_Handler();
	}
#endif

}

char th_getchar() { return getchar(); }

void th_serialport_initialize(void)
{

}

void th_timestamp(void)
{
#if EE_CFG_ENERGY_MODE == 1
/* USER CODE 1 BEGIN */
/* Step 1. Pull pin low */
	clear_Timestamp_Port();
/* Step 2. Hold low for at least 1us */
	US_Delay(1);
	set_Timestamp_Port();
/* Step 3. Release driver */
/* USER CODE 1 END */
#else

    /* USER CODE 2 BEGIN */
	unsigned long microSeconds = __HAL_TIM_GET_COUNTER(&htim5);
    /* USER CODE 2 END */
    /* This message must NOT be changed. */
    th_printf(EE_MSG_TIMESTAMP, microSeconds);
#endif
}

void th_timestamp_initialize(void)
{
    th_printf(EE_MSG_TIMESTAMP_MODE);
    th_timestamp();

}
