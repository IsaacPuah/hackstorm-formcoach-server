#include "tuya_cloud_types.h"
#include <assert.h>
#include "cJSON.h"
#include "tal_api.h"
#include "tuya_config.h"
#include "tuya_iot.h"
#include "netmgr.h"
#include "tkl_output.h"
#include "tuya_authorize.h"
#include "netconn_wifi.h"
#if defined(ENABLE_LIBLWIP) && (ENABLE_LIBLWIP == 1)
#include "lwip_init.h"
#endif
#include "board_com_api.h"
#include "http_client_interface.h"
#include "tkl_network.h"
#include "ai_video_input.h"
#include "app_chat_bot.h"
#include "ai_ui_manage.h"

#define WIFI_SSID     "ATTQ4uqgQB"
#define WIFI_PASSWORD "t8imckw8evn4"
#define SERVER_IP     "192.168.1.122"
#define SERVER_PORT   5001
#define SERVER_PATH   "/analyze"

tuya_iot_client_t ai_client;
tuya_iot_license_t license;

#ifndef PROJECT_VERSION
#define PROJECT_VERSION "1.0.0"
#endif

static void formcoach_http_test(void)
{
    PR_INFO("FormCoach: Waiting for WiFi...");
    int wait_count = 0;
    while (wait_count < 30) {
        netmgr_status_e status = NETMGR_LINK_DOWN;
        netmgr_conn_get(NETCONN_AUTO, NETCONN_CMD_STATUS, &status);
        if (status != NETMGR_LINK_DOWN) break;
        tal_system_sleep(1000);
        wait_count++;
    }
    tal_system_sleep(2000);

    while (1) {
        PR_INFO("FormCoach: Capturing camera frame...");
        
        uint8_t *jpeg_data = NULL;
        uint32_t jpeg_len = 0;
        OPERATE_RET cam_ret = ai_video_get_jpeg_frame(&jpeg_data, &jpeg_len);
        
        if (cam_ret == OPRT_OK && jpeg_data != NULL && jpeg_len > 0) {
            PR_INFO("FormCoach: Captured JPEG, size: %d bytes", jpeg_len);
            
            // Build header
            char header[256];
            int header_len = snprintf(header, sizeof(header),
                "POST %s HTTP/1.1\r\n"
                "Host: %s:%d\r\n"
                "Content-Type: image/jpeg\r\n"
                "Content-Length: %d\r\n"
                "Connection: close\r\n"
                "\r\n",
                SERVER_PATH, SERVER_IP, SERVER_PORT, (int)jpeg_len);

            int fd = tkl_net_socket_create(PROTOCOL_TCP);
            if (fd >= 0) {
                TUYA_IP_ADDR_T server_ip = tkl_net_str2addr(SERVER_IP);
                int ret = tkl_net_connect(fd, server_ip, SERVER_PORT);
                if (ret >= 0) {
                    tkl_net_send(fd, header, header_len);
                    tkl_net_send(fd, (const char *)jpeg_data, jpeg_len);

                    char resp[2048] = {0};
                    int total = 0, chunk;
                    while ((chunk = tkl_net_recv(fd, resp + total, sizeof(resp) - total - 1)) > 0) {
                        total += chunk;
                        if (total >= sizeof(resp) - 1) break;
                    }
                    resp[total] = '\0';

                    char *body = strstr(resp, "\r\n\r\n");
                    if (body) {
                        body += 4;
                        char *label_start = strstr(body, "\"label\":\"");
                        if (label_start) {
                            label_start += 9;
                            char label[32] = {0};
                            int i = 0;
                            while (label_start[i] && label_start[i] != '"' && i < 31) {
                                label[i] = label_start[i];
                                i++;
                            }
                            label[i] = '\0';
                            PR_INFO("FormCoach: Label = %s", label);
                            ai_ui_disp_msg(AI_UI_DISP_NOTIFICATION, (uint8_t *)label, strlen(label));
                        }
                    }
                }
                tkl_net_close(fd);
                ai_video_jpeg_image_free(&jpeg_data);
            }
        }
        
        tal_system_sleep(100);
    }

    PR_INFO("FormCoach: Capturing camera frame...");

    // Capture JPEG from camera
    uint8_t *jpeg_data = NULL;
    uint32_t jpeg_len = 0;
    OPERATE_RET cam_ret = ai_video_get_jpeg_frame(&jpeg_data, &jpeg_len);
    if (cam_ret != OPRT_OK || jpeg_data == NULL || jpeg_len == 0) {
        PR_ERR("FormCoach: camera capture failed: %d", cam_ret);
        // Fall back to test data
        jpeg_data = (uint8_t *)"{\"test\":\"hello\"}";
        jpeg_len = strlen((char *)jpeg_data);
    } else {
        PR_INFO("FormCoach: Captured JPEG, size: %d bytes", jpeg_len);
    }

    // Build HTTP header
    char header[256];
    int header_len = snprintf(header, sizeof(header),
        "POST %s HTTP/1.1\r\n"
        "Host: %s:%d\r\n"
        "Content-Type: image/jpeg\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n",
        SERVER_PATH, SERVER_IP, SERVER_PORT, (int)jpeg_len);

    // Connect via raw TCP
    int fd = tkl_net_socket_create(PROTOCOL_TCP);
    if (fd < 0) {
        PR_ERR("FormCoach: socket create failed");
        return;
    }

    TUYA_IP_ADDR_T server_ip = tkl_net_str2addr(SERVER_IP);
    int ret = tkl_net_connect(fd, server_ip, SERVER_PORT);
    if (ret < 0) {
        PR_ERR("FormCoach: connect failed: %d", ret);
        tkl_net_close(fd);
        return;
    }

    // Send header first
    ret = tkl_net_send(fd, header, header_len);
    if (ret < 0) {
        PR_ERR("FormCoach: header send failed: %d", ret);
        tkl_net_close(fd);
        return;
    }

    // Send JPEG body
    ret = tkl_net_send(fd, (const char *)jpeg_data, jpeg_len);
    if (ret < 0) {
        PR_ERR("FormCoach: body send failed: %d", ret);
        tkl_net_close(fd);
        return;
    }

    PR_INFO("FormCoach: Frame sent! Waiting for response...");

    // Read response
    // Read full response including body
    char resp[2048] = {0};
    int total = 0;
    int chunk;
    while ((chunk = tkl_net_recv(fd, resp + total, sizeof(resp) - total - 1)) > 0) {
        total += chunk;
        if (total >= sizeof(resp) - 1) break;
    }
    resp[total] = '\0';

    if (total > 0) {
        // Find JSON body after headers
        char *body = strstr(resp, "\r\n\r\n");
        if (body) {
            body += 4;
            PR_INFO("FormCoach: JSON body: %s", body);

            // Parse label
            char *label_start = strstr(body, "\"label\":\"");
            if (label_start) {
                label_start += 9;
                char label[32] = {0};
                int i = 0;
                while (label_start[i] && label_start[i] != '"' && i < 31) {
                    label[i] = label_start[i];
                    i++;
                }
                label[i] = '\0';
                PR_INFO("FormCoach: Label = %s", label);
                ai_ui_disp_msg(AI_UI_DISP_NOTIFICATION, (uint8_t *)label, strlen(label));
            }
        }
    }

    tkl_net_close(fd);
    PR_INFO("FormCoach: Done!");
}

bool user_network_check(void)
{
    netmgr_status_e status = NETMGR_LINK_DOWN;
    netmgr_conn_get(NETCONN_AUTO, NETCONN_CMD_STATUS, &status);
    return status == NETMGR_LINK_DOWN ? false : true;
}

void user_event_handler_on(tuya_iot_client_t *client, tuya_event_msg_t *event)
{
    PR_DEBUG("Tuya Event ID:%d(%s)", event->id, EVENT_ID2STR(event->id));
    switch (event->id) {
    case TUYA_EVENT_BIND_START:
        PR_INFO("WiFi Connected! Starting FormCoach HTTP test...");
        formcoach_http_test();
        break;
    default:
        break;
    }
}

void user_main(void)
{
    OPERATE_RET ret = OPRT_OK;

    tal_log_init(TAL_LOG_LEVEL_DEBUG, 1024, (TAL_LOG_OUTPUT_CB)tkl_log_output);
    PR_NOTICE("=== FormCoach Firmware Starting ===");

    tal_kv_init(&(tal_kv_cfg_t){
        .seed = "vmlkasdh93dlvlcy",
        .key  = "dflfuap134ddlduq",
    });
    tal_sw_timer_init();
    tal_workq_init();
    tal_time_service_init();
    tuya_authorize_init();

    if (OPRT_OK != tuya_authorize_read(&license)) {
        license.uuid    = TUYA_OPENSDK_UUID;
        license.authkey = TUYA_OPENSDK_AUTHKEY;
    }

    ret = tuya_iot_init(&ai_client, &(const tuya_iot_config_t){
        .software_ver  = PROJECT_VERSION,
        .productkey    = TUYA_PRODUCT_ID,
        .uuid          = license.uuid,
        .authkey       = license.authkey,
        .event_handler = user_event_handler_on,
        .network_check = user_network_check,
    });
    assert(ret == OPRT_OK);

#if defined(ENABLE_LIBLWIP) && (ENABLE_LIBLWIP == 1)
    TUYA_LwIP_Init();
#endif

    netmgr_init(NETCONN_WIFI);

    netconn_wifi_info_t wifi_info = {0};
    strncpy(wifi_info.ssid, WIFI_SSID, sizeof(wifi_info.ssid) - 1);
    strncpy(wifi_info.pswd, WIFI_PASSWORD, sizeof(wifi_info.pswd) - 1);
    netmgr_conn_set(NETCONN_WIFI, NETCONN_CMD_SSID_PSWD, &wifi_info);

    PR_INFO("Connecting to WiFi: %s", WIFI_SSID);

    ret = board_register_hardware();
    if (ret != OPRT_OK) {
        PR_ERR("board_register_hardware failed");
    }
    
    ret = app_chat_bot_init();
    if (ret != OPRT_OK) {
        PR_ERR("app_chat_bot_init failed");
    }

    tuya_iot_start(&ai_client);

    for (;;) {
        tuya_iot_yield(&ai_client);
        tal_system_sleep(100);
    }
}

static THREAD_HANDLE ty_app_thread = NULL;

static void tuya_app_thread(void *arg)
{
    user_main();
    tal_thread_delete(ty_app_thread);
    ty_app_thread = NULL;
}

void tuya_app_main(void)
{
    THREAD_CFG_T thrd_param = {0};
    thrd_param.stackDepth   = 4096;
    thrd_param.priority     = 4;
    thrd_param.thrdname     = "tuya_app_main";
    tal_thread_create_and_start(&ty_app_thread, NULL, NULL, tuya_app_thread, NULL, &thrd_param);
}