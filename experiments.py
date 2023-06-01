from noise_reduction import *

if __name__ == "__main__":
    # 画像のパスとフィルタのパラメータを指定する
    original_path = 'img/original_image.jpg'
    image_path = 'img/noisy_image_gaussian.jpg'

    original_image = cv2.imread(original_path, 0)
    noisy_image = cv2.imread(image_path, 0)

    print(psnr(original_image, noisy_image))
    rs = [2, 3, 4, 5]
    sigma_i = [1.0, 5.0, 10.0, 25.0, 40.0]
    sigma_s = [1.0, 5.0, 10.0, 25.0, 40.0]

    ws = [3, 6, 9]
    epss = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.1]
    hs = [250, 500, 750, 1000]
    # eps = 0.23**2  # epsilon
    

    bilateral_result = []
    nlm_result = []
    guided_result = []

    # bilateral filter 
    # for r in rs:
    #     for s in sigma_i:
    #         for t in sigma_s:
    #             print(f'r: {r}, sigma_i: {s}, sigma_s: {t}')

    #             filtered_image = bilateral_filter(noisy_image, 2*r+1, s, t)
    #             cv2.imwrite(f'img/bilateral_filter_{r}_{s}_{t}.jpg', filtered_image)
    #             print(f'psnr: {psnr(original_image, filtered_image)}')
    #             bilateral_result.append({"r": r, "sigma_i": s, "sigma_s": t, "psnr": psnr(original_image, filtered_image)})

    # # save psnr result
    # df = pd.DataFrame(bilateral_result)
    # df.to_csv("bilateral_filter_psnr.csv")


    
    
    # non-local mean filter
    # for r in rs:
    #     for w in ws:
    #         for h in hs:
    #             print(f'r: {r}, w: {w}, h: {h}')
    #             filtered_image = non_local_means(noisy_image, r, h, w)
    #             cv2.imwrite(f'img/non_local_means_{r}_{w}_{h}.jpg', filtered_image)
    #             print(f'psnr: {psnr(original_image, filtered_image)}')
    #             nlm_result.append({"r": r, "h": h, "psnr": psnr(original_image, filtered_image)})

    # for r in rs:
    #     for w in ws:
    #         for h in hs:
    #             image = cv2.imread(f"img/non_local_means_{r}_{w}_{h}.jpg", 0)
    #             nlm_result.append({"r": r, "w": w, "h": h, "psnr": psnr(original_image, image)})
                
    # df = pd.DataFrame(nlm_result)
    # df.to_csv("non_local_means_psnr.csv")
    # guided filter
    for r in rs:
        for eps in epss:
            filtered_image = process_image(noisy_image, r, eps)
            cv2.imwrite(f'img/guided_filter_{r}_{eps}.jpg', filtered_image)
            print(f'psnr: {psnr(original_image, filtered_image)}')
            guided_result.append({"r": r, "eps": eps, "psnr": psnr(original_image, filtered_image)})


    
    
   

    df = pd.DataFrame(guided_result)
    df.to_csv("guided_filter_psnr.csv")

    
