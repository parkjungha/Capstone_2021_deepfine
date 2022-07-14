#record time taken being transformed
    duration=[] #duration은 이미지 백장에 대한 time을 담기 위한 배열
    starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
#GPU-WARM-UP
    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b0')
    device = torch.device("cuda")
    model.to(device)
    dummy_input = torch.randn(1, 3,224,224,dtype=torch.float).to(device)
    for _ in range(10):
       _ = model(dummy_input)
    while True:
        starter.record()
        #이미지 변환 및 내보내는 코드
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        #i=0일 때는 GPU-WARM-UP
        if i>0 and i<=100:
        	elapsed_time=starter.elapsed_time(ender) #elapsed_time이 이미지 변환하는 데 걸린 시간
        	duration.append(elapsed_time) # duration 배열에 담음
        i+=1
        if i==101:
        	print(round(sum(duration)/len(duration))) #이미지 백장에 대한 time 총합을 이미지 개수로 나눈 값 출력
        	break
