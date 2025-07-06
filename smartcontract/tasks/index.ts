import { task } from "hardhat/config";

// Deploy Subscription contract
task("deploy-subscription").setAction(async (_args, hre) => {
  const Subscription = await hre.ethers.getContractFactory("Subscription");
  const subscription = await Subscription.deploy();
  const subscriptionAddr = await subscription.waitForDeployment();

  console.log(`Subscription contract deployed to: ${subscriptionAddr.target}`);
  return subscriptionAddr.target;
});

// Subscribe to the service
task("subscribe")
  .addParam("address", "contract address")
  .setAction(async (args, hre) => {
    const subscription = await hre.ethers.getContractAt("Subscription", args.address);
    const monthlyPrice = await subscription.monthlyPrice();
    
    console.log(`Monthly price: ${hre.ethers.formatEther(monthlyPrice)} TEST tokens`);
    
    const tx = await subscription.subscribe({ value: monthlyPrice });
    await tx.wait();
    
    console.log("✅ Subscription successful!");
    console.log(`Transaction hash: ${tx.hash}`);
  });

// Check subscription status
task("check-subscription")
  .addParam("address", "contract address")
  .addOptionalParam("user", "user address to check (defaults to current signer)")
  .setAction(async (args, hre) => {
    const subscription = await hre.ethers.getContractAt("Subscription", args.address);
    const [signer] = await hre.ethers.getSigners();
    const userAddress = args.user || signer.address;
    
    console.log(`Checking subscription for: ${userAddress}`);
    
    const isActive = await subscription.isSubscriptionActive(userAddress);
    const subData = await subscription.getSubscription(userAddress);
    const remainingDays = await subscription.getRemainingDays(userAddress);
    
    console.log(`Is Active: ${isActive}`);
    console.log(`Subscription Date: ${new Date(Number(subData[0]) * 1000).toLocaleString()}`);
    console.log(`Expiry Date: ${new Date(Number(subData[1]) * 1000).toLocaleString()}`);
    console.log(`Monthly Price: ${hre.ethers.formatEther(subData[2])} TEST`);
    console.log(`Remaining Days: ${remainingDays}`);
  });

// Cancel subscription
task("cancel-subscription")
  .addParam("address", "contract address")
  .setAction(async (args, hre) => {
    const subscription = await hre.ethers.getContractAt("Subscription", args.address);
    
    const tx = await subscription.cancelSubscription();
    await tx.wait();
    
    console.log("✅ Subscription cancelled!");
    console.log(`Transaction hash: ${tx.hash}`);
  });

// Full subscription test
task("test-subscription").setAction(async (_args, hre) => {
  await hre.run("compile");
  
  const address = await hre.run("deploy-subscription");
  
  console.log("\n--- Testing Subscription ---");
  await hre.run("subscribe", { address });
  await hre.run("check-subscription", { address });
  
  console.log("\n--- Test Complete ---");
}); 