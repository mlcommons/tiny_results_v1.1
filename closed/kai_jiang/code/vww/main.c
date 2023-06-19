
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "drv_usart.h"
#include "timer.h"
#include "stdio.h"
#include "soc.h"
#include "pin.h"
#include "app.h"
#include "internally_implemented.h"
#include "submitter_implemented.h"
	
extern usart_handle_t console_handle;
extern uint8_t gp_buff[MAX_DB_INPUT_SIZE]__attribute__((section(".dtcm")));
static volatile uint8_t rx_async_flag = 0;
static volatile uint8_t tx_async_flag = 0;
static volatile uint8_t rx_trigger_flag = 0;
static volatile uint8_t rx_flag = 0;
static uint16_t read_cnt = 0;
uint8_t g_ctl = 0;
uint64_t g_time_start = 0;
uint32_t data_ready = 0;
uint32_t read_data = 0;

void delay_us(uint32_t time)
{
	uint32_t n;
	for(n = 0;n<time; n++)
	{
		__NOP();
		__NOP();
		__NOP();
		__NOP();
		__NOP();
		__NOP();
		__NOP();
		__NOP();
		__NOP();
		__NOP();
		__NOP();
		__NOP();
		__NOP();
		__NOP();
		__NOP();
		__NOP();
		__NOP();
		__NOP();
	}
}
uint32_t get_current_clock(void)
{
	uint64_t ntime;
	
	ntime = SysTimer_GetLoadValue();
	if(ntime > g_time_start)
	{
		return((ntime - g_time_start)/syscalc);
	}
	return 0;
}
void readdata(void)
{
	uint16_t byIndex = 0;
	uint16_t bywrite = 0;
	uint16_t byLen = 0;
	
	bywrite = g_Serial->wp;
	if(g_Serial->rp != bywrite)
	{
		byLen =  (bywrite + CH_LEN - g_Serial->rp)%CH_LEN;

		/* 拷贝数据 */
		for(byIndex=0; byIndex<byLen; byIndex++)
		{
			ee_serial_callback(g_Serial->addr[g_Serial->rp]);
			if(++g_Serial->rp >= CH_LEN)
				g_Serial->rp = 0;
		}
	}
}
static void usart_event_cb(int32_t idx, uint32_t event)
{
	char g_data[64];
	uint16_t uwNum = 0;
	uint16_t i;

	switch (event) {
		case USART_EVENT_SEND_COMPLETE:
		 tx_async_flag = 1;
			break;
		case USART_EVENT_RECEIVE_COMPLETE:
		 rx_async_flag = 1;
			break;
		case USART_EVENT_RECEIVED:
			uwNum = csi_usart_receive_query(console_handle, g_data, 64);
			read_cnt = 0;
			for(i=0;i<uwNum;i++)
			{
				g_recvbuf[g_Serial->wp] = g_data[i];
				if(++g_Serial->wp >= CH_LEN)
					g_Serial->wp = 0;
			}
			rx_flag = 1;

		default:
			break;
	}
}
void Uart_init(void)
{
	g_Serial = &SerStruct;	
	g_Serial->addr = g_recvbuf;
	g_Serial->rp = 0;
	g_Serial->wp = 0;
}


int main(void)
{

	syscalc = drv_get_sys_freq()/1000000;//us
	if(syscalc==0)
		syscalc = 20000;
	Uart_init();
	console_handle = csi_usart_initialize(CONSOLE_IDX, (usart_event_cb_t)usart_event_cb);
    /* config the UART */
    csi_usart_config(console_handle, 115200, USART_MODE_ASYNCHRONOUS, USART_PARITY_NONE, USART_STOP_BITS_1, USART_DATA_BITS_8);

	write_ai_cmd();
	write_ai_para();
	read_ai_status();
	g_time_start = SysTimer_GetLoadValue();
	ee_benchmark_initialize();

	while(1)
	{
		readdata();
	}

    return 0;
}
